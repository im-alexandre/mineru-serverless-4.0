import asyncio
import hashlib
import ipaddress
import os
import posixpath
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

import boto3
import httpx
import requests
import runpod
import torch
from botocore.config import Config
from mineru.parser import MinerUApiParser
from mineru.parser.writer import FileBasedDataWriter

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ROUTER_HOST = "127.0.0.1"
ROUTER_PORT = int(os.environ.get("MINERU_ROUTER_PORT", "8002"))
ROUTER_URL = f"http://{ROUTER_HOST}:{ROUTER_PORT}"

MINERU_TIER = os.environ.get("MINERU_TIER", "standard")
CONCURRENCY_PER_GPU = int(os.environ.get("MINERU_CONCURRENCY_PER_GPU", "3"))

ROUTER_STARTUP_TIMEOUT = int(os.environ.get("MINERU_ROUTER_STARTUP_TIMEOUT", "900"))

S3_ENDPOINT_URL = os.environ["S3_ENDPOINT_URL"]
S3_ACCESS_KEY_ID = os.environ["S3_ACCESS_KEY_ID"]
S3_SECRET_ACCESS_KEY = os.environ["S3_SECRET_ACCESS_KEY"]
S3_REGION = os.environ.get("S3_REGION", "us-east-1")


# ---------------------------------------------------------------------
# GPU / capacity
# ---------------------------------------------------------------------

GPU_COUNT = torch.cuda.device_count()

if GPU_COUNT <= 0:
    raise RuntimeError("No CUDA GPUs available")

MAX_WORKER_CONCURRENCY = GPU_COUNT * CONCURRENCY_PER_GPU


print(
    f"[worker] GPUs={GPU_COUNT} "
    f"concurrency_per_gpu={CONCURRENCY_PER_GPU} "
    f"max_concurrency={MAX_WORKER_CONCURRENCY}",
    flush=True,
)


# ---------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT_URL,
    aws_access_key_id=S3_ACCESS_KEY_ID,
    aws_secret_access_key=S3_SECRET_ACCESS_KEY,
    region_name=S3_REGION,
    config=Config(
        signature_version="s3v4",
        retries={
            "max_attempts": 5,
            "mode": "standard",
        },
    ),
)


# ---------------------------------------------------------------------
# MinerU Router
# ---------------------------------------------------------------------

_router_process: subprocess.Popen | None = None


def router_health() -> dict | None:
    try:
        response = httpx.get(
            f"{ROUTER_URL}/v1/health",
            timeout=5,
        )

        if response.status_code != 200:
            return None

        return response.json()

    except Exception:
        return None


def start_router() -> None:
    global _router_process

    if router_health():
        return

    command = [
        "mineru-kit",
        "router",
        "--host",
        ROUTER_HOST,
        "--port",
        str(ROUTER_PORT),
        "--local-gpus",
        "auto",
        "--worker-tier",
        MINERU_TIER,
        "--worker-concurrency",
        str(CONCURRENCY_PER_GPU),
        "--preload-models",
    ]

    print(
        "[worker] starting MinerU router:",
        " ".join(command),
        flush=True,
    )

    _router_process = subprocess.Popen(
        command,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    deadline = time.monotonic() + ROUTER_STARTUP_TIMEOUT

    while time.monotonic() < deadline:
        health = router_health()

        if health:
            print(
                f"[worker] MinerU router ready: {health}",
                flush=True,
            )
            return

        if _router_process.poll() is not None:
            raise RuntimeError(
                "MinerU router exited during startup "
                f"with code {_router_process.returncode}"
            )

        time.sleep(2)

    raise TimeoutError("MinerU router did not become healthy")


_router_lock = threading.Lock()


def ensure_router() -> None:
    """Restart the router if it has died since startup.

    Called before every job because start_router() only guarantees
    health at import time; a crash afterward would otherwise fail
    every subsequent job with no recovery.
    """
    if router_health():
        return

    with _router_lock:
        if router_health():
            return

        print(
            "[worker] MinerU router is unreachable, restarting",
            flush=True,
        )
        start_router()


# Start once when RunPod creates this worker.
start_router()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def source_filename(url: str) -> str:
    path = unquote(urlparse(url).path)
    filename = Path(path).name

    return filename or "document.pdf"


def output_key(
    filename: str,
    requested_key: str,
) -> str:
    """Resolve the destination S3 key for a job's output tarball.

    ``requested_key`` is treated as a directory prefix when it ends
    in "/" (or is empty); the tarball is named after the source PDF.
    Anything else is treated as the literal final key the caller
    wants, unchanged.
    """

    requested_key = requested_key.lstrip("/")
    tarball_name = f"{Path(filename).stem}.tar.gz"

    if not requested_key or requested_key.endswith("/"):
        prefix = requested_key.rstrip("/")

        if prefix:
            return posixpath.join(
                prefix,
                tarball_name,
            )

        return tarball_name

    return requested_key


class UnsafeURLError(ValueError):
    """Raised when a job-supplied URL is not safe to fetch."""


_ALLOWED_URL_SCHEMES = {"http", "https"}


def _is_public_address(address: str) -> bool:
    ip = ipaddress.ip_address(address)

    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def assert_safe_url(url: str) -> None:
    """Reject job-supplied URLs that could be used for SSRF.

    Job input is attacker-controlled (it names the URL this worker
    downloads), so we only allow http(s) URLs whose host resolves
    exclusively to public IPs, rejecting cloud metadata endpoints
    and other internal services.
    """
    parsed = urlparse(url)

    if parsed.scheme not in _ALLOWED_URL_SCHEMES:
        raise UnsafeURLError(f"unsupported URL scheme: {parsed.scheme!r}")

    if not parsed.hostname:
        raise UnsafeURLError("URL has no host")

    try:
        resolved = {
            info[4][0] for info in socket.getaddrinfo(parsed.hostname, None)
        }
    except socket.gaierror as exc:
        raise UnsafeURLError(
            f"could not resolve host: {parsed.hostname!r}"
        ) from exc

    if not resolved or not all(_is_public_address(ip) for ip in resolved):
        raise UnsafeURLError(
            f"host resolves to a non-public address: {parsed.hostname!r}"
        )


def download_file(
    url: str,
    destination: Path,
) -> None:

    assert_safe_url(url)

    with requests.get(
        url,
        stream=True,
        timeout=(30, 1800),
        allow_redirects=False,
    ) as response:
        if 300 <= response.status_code < 400:
            raise UnsafeURLError("redirects are not followed for job-supplied URLs")

        response.raise_for_status()

        with destination.open("wb") as file:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    file.write(chunk)


def create_tarball(
    source_dir: Path,
    destination: Path,
) -> None:

    with tarfile.open(
        destination,
        "w:gz",
    ) as tar:
        for item in source_dir.iterdir():
            tar.add(
                item,
                arcname=item.name,
                recursive=True,
            )


def sha256(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


# ---------------------------------------------------------------------
# Synchronous job implementation
# ---------------------------------------------------------------------


def process_job(job: dict) -> dict:

    ensure_router()

    data = job["input"]

    url = data["url"]

    tier = data.get(
        "tier",
        MINERU_TIER,
    )

    pages = data.get(
        "pages",
        "all",
    )

    destination = data["destination"]

    bucket = destination["bucket"]
    requested_key = destination["key"]

    filename = source_filename(url)

    key = output_key(
        filename,
        requested_key,
    )

    with tempfile.TemporaryDirectory(prefix="mineru-") as tmpdir:
        tmp = Path(tmpdir)

        input_path = tmp / filename
        output_dir = tmp / "output"

        tarball_path = tmp / f"{Path(filename).stem}.tar.gz"

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        download_file(
            url,
            input_path,
        )

        #
        # Router distributes this request to one
        # of its per-GPU api-server workers.
        #
        local_parser = MinerUApiParser(
            api_url=ROUTER_URL,
            api_key="",
            tier=tier,
            include_images=True,
        )

        result = local_parser.parse(
            str(input_path),
            page_range=pages,
        )

        #
        # Complete MinerU package:
        #
        # markdown.md
        # middle_json.json
        # structured_content.json
        # model_output.json (when available)
        # images/
        #
        result.save(FileBasedDataWriter(str(output_dir)))

        create_tarball(
            output_dir,
            tarball_path,
        )

        size = tarball_path.stat().st_size
        checksum = sha256(tarball_path)

        s3.upload_file(
            str(tarball_path),
            bucket,
            key,
            ExtraArgs={
                "ContentType": "application/gzip",
                "Metadata": {
                    "source-filename": filename,
                    "mineru-tier": tier,
                    "sha256": checksum,
                },
            },
        )

        return {
            "status": "completed",
            "source": filename,
            "pages": len(result.pages),
            "tier": tier,
            "bucket": bucket,
            "key": key,
            "size": size,
            "sha256": checksum,
        }


# ---------------------------------------------------------------------
# RunPod async handler
# ---------------------------------------------------------------------


async def handler(job):

    #
    # process_job() is blocking.
    #
    # to_thread allows several RunPod jobs to coexist
    # in the same worker while MinerU Router controls
    # GPU inference concurrency.
    #
    return await asyncio.to_thread(
        process_job,
        job,
    )


# ---------------------------------------------------------------------
# RunPod
# ---------------------------------------------------------------------

if __name__ == "__main__":
    runpod.serverless.start(
        {
            "handler": handler,
            #
            # Example:
            #
            # 1 GPU × 3 = 3
            # 2 GPUs × 3 = 6
            # 4 GPUs × 3 = 12
            #
            "concurrency_modifier": (lambda current: MAX_WORKER_CONCURRENCY),
        }
    )
