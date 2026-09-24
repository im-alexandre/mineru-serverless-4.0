# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A RunPod serverless worker that wraps [MinerU](https://github.com/opendatalab/MinerU) (PDF parsing/OCR with a vLLM-backed VLM) behind a GPU worker. A client submits a PDF URL + S3 destination, the worker downloads the PDF, runs it through a local MinerU router, tars the output package, and uploads the tarball to S3-compatible storage.

There is no test suite, linter, or local build step configured — this repo is a Docker image + two Python entrypoints.

## Architecture

Three files carry all the logic:

- **`handler.py`** — runs *inside* the RunPod worker container (GPU host). On import it starts a local `mineru-kit router` subprocess (`MINERU_ROUTER_PORT`, default 8002) that owns GPU inference and load-balances across per-GPU API workers, and blocks until `/v1/health` responds. `process_job()` does the actual work per request: download the source PDF from a URL, run `MinerUApiParser.parse()` against the local router, save the MinerU output package (`markdown.md`, `middle_json.json`, `structured_content.json`, optional `model_output.json`, `images/`) with `FileBasedDataWriter`, tar it, and upload to S3 with a sha256 in the object metadata. `concurrency_modifier` is `GPU_COUNT * MINERU_CONCURRENCY_PER_GPU` so RunPod knows how many concurrent jobs one worker can host — actual GPU scheduling concurrency is still controlled by the router, not by this modifier.
- **`client.py`** — runs *outside* RunPod, on the caller's side. Uploads a local PDF to S3, generates a presigned GET URL, and submits `{url, tier, pages, destination: {bucket, key}}` to the RunPod endpoint via the `runpod` SDK. Includes `run_batch()`, which submits a queue of documents while dynamically sizing in-flight job count off live endpoint health (`get_capacity()` reads `workers.idle/running/ready/initializing` and `jobs.inQueue/inProgress` from RunPod's `/health` endpoint) rather than a fixed concurrency number — this is how the client avoids over-queuing beyond what warm workers can absorb.
- **`process.py`** — a scratch/example script (hardcoded endpoint ID and placeholder SeaweedFS credentials) showing a single `run_sync` call. Not part of the production path; treat as a usage example, not something to import from.

### Job contract (handler input/output)

Request (`job["input"]`):
```json
{
  "url": "<presigned or public PDF URL>",
  "tier": "standard",       // optional, defaults to MINERU_TIER
  "pages": "all",           // optional, MinerU page-range syntax
  "destination": {"bucket": "...", "key": "prefix/or/exact/key.tar.gz"}
}
```
`output_key()` derives the final tarball name from the source filename when `destination.key` ends in `/` (or is empty); otherwise it uses the exact key supplied. The client sends `S3_OUTPUT_PREFIX/`, so the output is `<prefix>/<PDF stem>.tar.gz`. The generated document ID is used only for the temporary input upload.

Response: `{status, source, pages, tier, bucket, key, size, sha256}`.

### Config (env vars)

Worker side (`handler.py`, set at image build/run time — see `Dockerfile` defaults): `MINERU_ROUTER_PORT`, `MINERU_TIER`, `MINERU_CONCURRENCY_PER_GPU`, `MINERU_ROUTER_STARTUP_TIMEOUT`, plus required `S3_ENDPOINT_URL` / `S3_ACCESS_KEY_ID` / `S3_SECRET_ACCESS_KEY` (optional `S3_REGION`).

Client side (`client.py`, loaded via `.env` / `python-dotenv`): required `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`, `S3_ENDPOINT_URL`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, `S3_BUCKET`, `S3_INPUT_PREFIX`, `S3_OUTPUT_PREFIX`; optional `RUNPOD_GPUS_PER_WORKER`, `MINERU_CONCURRENCY_PER_GPU`, `RUNPOD_POLL_INTERVAL`, `RUNPOD_JOB_TIMEOUT`, `RUNPOD_AUTOSCALE_QUEUE_BUFFER`, `S3_REGION`, `S3_INPUT_URL_EXPIRES`.

Both sides read S3 credentials independently — worker and client can point at different S3-compatible endpoints/buckets.

## Working with this repo

- Dependency manifests are split: `pyproject.toml`/`uv.lock` (requires Python >=3.14, managed with `uv`) covers local/client-side dependencies including `python-dotenv`; `requirements.txt` and the `Dockerfile`'s own `pip install` are what actually ship inside the worker image (`mineru[torch]==4.0.7`, `runpod`, `requests`, `httpx`, `boto3`) and do **not** include `python-dotenv` — `handler.py` reads env vars directly with no `.env` loading, since RunPod injects them into the container.
- The Docker image is CUDA/GPU-only: `handler.py` calls `torch.cuda.device_count()` at import time and raises immediately if no GPU is visible, so it cannot be run or imported on a CPU-only machine.
- Building the image bakes in MinerU's model weights (`mineru-kit models download --tier standard ... --source huggingface`) so worker startup doesn't need to fetch them; changing `MINERU_TIER`/backend/engine defaults means updating the matching `mineru-kit models download`/`verify` flags in the `Dockerfile` too.
- CI (`.github/workflows/build.yml`) builds and pushes the image to `ghcr.io/<owner>/mineru-runpod-worker` on every push to `main` — there's no test/lint job in the pipeline.
- To test client-side logic without RunPod credentials, exercise `client.py`'s pure helpers (`upload_pdf`, `output_key`-equivalent logic lives in `handler.py`, `get_capacity`) rather than `submit_document`/`run_batch`, which require live `RUNPOD_API_KEY`/`RUNPOD_ENDPOINT_ID` and a reachable endpoint.
