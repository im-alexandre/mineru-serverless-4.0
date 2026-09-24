import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx
import requests
import runpod
from mineru.parser import MinerUApiParser

MINERU_API_HOST = "127.0.0.1"
MINERU_API_PORT = int(os.environ.get("MINERU_API_PORT", "8000"))
MINERU_API_URL = f"http://{MINERU_API_HOST}:{MINERU_API_PORT}"
MINERU_SERVER_TIER = os.environ.get("MINERU_SERVER_TIER", "standard")
MINERU_SERVER_STARTUP_TIMEOUT = float(os.environ.get("MINERU_SERVER_STARTUP_TIMEOUT", "600"))

_server_lock = threading.Lock()
_server_process: subprocess.Popen | None = None


def _server_healthy() -> bool:
    try:
        response = httpx.get(f"{MINERU_API_URL}/v1/health", timeout=3)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def ensure_mineru_server() -> None:
    """Start the local MinerU REST API server (mineru-api) once per worker process."""
    global _server_process

    if _server_healthy():
        return

    with _server_lock:
        if _server_healthy():
            return

        if _server_process is None or _server_process.poll() is not None:
            _server_process = subprocess.Popen(
                [
                    "mineru-api",
                    "--host",
                    MINERU_API_HOST,
                    "--port",
                    str(MINERU_API_PORT),
                    "--tier",
                    MINERU_SERVER_TIER,
                    "--allow-local-source",
                    "--preload-models",
                ],
                stdout=sys.stdout,
                stderr=sys.stderr,
            )

        deadline = time.monotonic() + MINERU_SERVER_STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if _server_healthy():
                return
            if _server_process.poll() is not None:
                raise RuntimeError(
                    f"mineru-api server exited unexpectedly during startup (code {_server_process.returncode})"
                )
            time.sleep(1)

        raise TimeoutError("mineru-api server did not become healthy in time")


def download_file(url: str, path: Path) -> None:
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()

        with path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def handler(job):
    input_data = job["input"]

    url = input_data["url"]
    tier = input_data.get("tier", MINERU_SERVER_TIER)
    pages = input_data.get("pages", "all")

    ensure_mineru_server()

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "input.pdf"

        download_file(url, pdf_path)

        parser = MinerUApiParser(api_url=MINERU_API_URL, tier=tier)
        result = parser.parse(pdf_path, page_range=pages)

        return {
            "pages": len(result.pages),
            "tier": tier,
            "markdown": result.markdown(),
        }


runpod.serverless.start({"handler": handler})
