import base64
import tarfile
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests
import runpod
from mineru.parser import parse
from mineru.parser.writer import FileBasedDataWriter

DEFAULT_TIER = "standard"


def download_file(url: str, dst: Path) -> None:
    with requests.get(url, stream=True, timeout=(30, 600)) as response:
        response.raise_for_status()

        with dst.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def filename_from_url(url: str) -> str:
    name = Path(urlparse(url).path).name

    if not name:
        return "input.pdf"

    return name


def create_tarball(source_dir: Path, tar_path: Path) -> None:
    with tarfile.open(tar_path, "w:gz") as tar:
        for item in source_dir.iterdir():
            tar.add(item, arcname=item.name)


def handler(job):
    data = job["input"]

    url = data["url"]
    tier = data.get("tier", DEFAULT_TIER)
    pages = data.get("pages", "all")
    ocr_mode = data.get("ocr_mode", "auto")

    with tempfile.TemporaryDirectory(prefix="mineru-") as tmpdir:
        tmp = Path(tmpdir)

        input_path = tmp / filename_from_url(url)
        output_dir = tmp / "output"
        tarball_path = tmp / "mineru-output.tar.gz"

        output_dir.mkdir(parents=True)

        download_file(url, input_path)

        result = parse(
            str(input_path),
            tier=tier,
            ocr_mode=ocr_mode,
            page_range=pages,
        )

        writer = FileBasedDataWriter(str(output_dir))
        result.save(writer)

        create_tarball(output_dir, tarball_path)

        tarball = tarball_path.read_bytes()

        return {
            "tier": tier,
            "pages": len(result.pages),
            "filename": "mineru-output.tar.gz",
            "size": len(tarball),
            "data_base64": base64.b64encode(tarball).decode("ascii"),
        }


runpod.serverless.start({"handler": handler})
