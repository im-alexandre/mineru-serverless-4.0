import tempfile
from pathlib import Path

import requests
import runpod
from mineru.parser import parse
from mineru.render import RenderFormat, render


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
    tier = input_data.get("tier", "standard")
    pages = input_data.get("pages", "all")

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "input.pdf"

        download_file(url, pdf_path)

        result = parse(
            pdf_path,
            tier=tier,
            ocr_mode="auto",
            page_range=pages,
        )

        markdown = render(
            result.middle_json,
            RenderFormat.MARKDOWN,
        )

        return {
            "pages": len(result.pages),
            "tier": tier,
            "markdown": markdown,
        }


runpod.serverless.start({"handler": handler})
