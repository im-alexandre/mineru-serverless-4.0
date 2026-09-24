import os
import time
import uuid
from pathlib import Path

import boto3
import requests
import runpod
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
RUNPOD_ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]

RUNPOD_GPUS_PER_WORKER = int(os.environ.get("RUNPOD_GPUS_PER_WORKER", "1"))

MINERU_CONCURRENCY_PER_GPU = int(os.environ.get("MINERU_CONCURRENCY_PER_GPU", "3"))

RUNPOD_POLL_INTERVAL = float(os.environ.get("RUNPOD_POLL_INTERVAL", "2"))

RUNPOD_JOB_TIMEOUT = int(os.environ.get("RUNPOD_JOB_TIMEOUT", "7200"))

RUNPOD_AUTOSCALE_QUEUE_BUFFER = int(
    os.environ.get(
        "RUNPOD_AUTOSCALE_QUEUE_BUFFER",
        str(RUNPOD_GPUS_PER_WORKER * MINERU_CONCURRENCY_PER_GPU),
    )
)

S3_ENDPOINT_URL = os.environ["S3_ENDPOINT_URL"]
S3_ACCESS_KEY_ID = os.environ["S3_ACCESS_KEY_ID"]
S3_SECRET_ACCESS_KEY = os.environ["S3_SECRET_ACCESS_KEY"]
S3_REGION = os.environ.get("S3_REGION", "us-east-1")

S3_BUCKET = os.environ["S3_BUCKET"]
S3_INPUT_PREFIX = os.environ["S3_INPUT_PREFIX"].strip("/")
S3_OUTPUT_PREFIX = os.environ["S3_OUTPUT_PREFIX"].strip("/")

S3_INPUT_URL_EXPIRES = int(os.environ.get("S3_INPUT_URL_EXPIRES", "3600"))


# ---------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------

runpod.api_key = RUNPOD_API_KEY

endpoint = runpod.Endpoint(
    RUNPOD_ENDPOINT_ID,
)

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
# S3
# ---------------------------------------------------------------------


def ensure_bucket() -> None:
    try:
        s3.head_bucket(Bucket=S3_BUCKET)
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in {"404", "NoSuchBucket", "NotFound"}:
            raise
        s3.create_bucket(Bucket=S3_BUCKET)


def upload_pdf(path: str | Path, document_id: str) -> tuple[str, str]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    ensure_bucket()

    key = f"{S3_INPUT_PREFIX}/{document_id}/{path.name}"

    s3.upload_file(
        str(path),
        S3_BUCKET,
        key,
        ExtraArgs={
            "ContentType": "application/pdf",
        },
    )

    presigned_url = s3.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": S3_BUCKET,
            "Key": key,
        },
        ExpiresIn=S3_INPUT_URL_EXPIRES,
    )

    return key, presigned_url


def delete_input(key: str) -> None:
    s3.delete_object(
        Bucket=S3_BUCKET,
        Key=key,
    )


# ---------------------------------------------------------------------
# RunPod health / capacity
# ---------------------------------------------------------------------


def get_endpoint_health() -> dict:
    response = requests.get(
        f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/health",
        headers={
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
        },
        timeout=10,
    )

    response.raise_for_status()

    return response.json()


def get_capacity() -> dict:
    health = get_endpoint_health()

    workers = health.get("workers", {})
    jobs = health.get("jobs", {})

    idle_workers = int(workers.get("idle", 0))

    running_workers = int(workers.get("running", 0))

    ready_workers = int(workers.get("ready", 0))

    initializing_workers = int(workers.get("initializing", 0))

    capacity_per_worker = RUNPOD_GPUS_PER_WORKER * MINERU_CONCURRENCY_PER_GPU

    warm_workers = max(
        ready_workers,
        idle_workers + running_workers,
    )

    warm_capacity = warm_workers * capacity_per_worker

    potential_capacity = (warm_workers + initializing_workers) * capacity_per_worker

    return {
        "idle_workers": idle_workers,
        "running_workers": running_workers,
        "ready_workers": ready_workers,
        "initializing_workers": initializing_workers,
        "capacity_per_worker": capacity_per_worker,
        "warm_capacity": warm_capacity,
        "potential_capacity": potential_capacity,
        "in_queue": int(jobs.get("inQueue", 0)),
        "in_progress": int(jobs.get("inProgress", 0)),
    }


# ---------------------------------------------------------------------
# RunPod jobs
# ---------------------------------------------------------------------


def normalize_status(job) -> str:
    status = job.status()

    if isinstance(status, dict):
        return status.get(
            "status",
            "",
        ).upper()

    return str(status).upper()


def submit_document(
    pdf_path: str | Path,
    *,
    tier: str = "standard",
    pages: str = "all",
):
    pdf_path = Path(pdf_path)

    document_id = uuid.uuid4().hex
    input_key, presigned_url = upload_pdf(pdf_path, document_id)

    payload = {
        "url": presigned_url,
        "tier": tier,
        "pages": pages,
        "destination": {
            "bucket": S3_BUCKET,
            "key": f"{S3_OUTPUT_PREFIX}/{document_id}/",
        },
    }

    job = endpoint.run(payload)

    return {
        "job": job,
        "input_key": input_key,
        "source": str(pdf_path),
        "payload": payload,
        "submitted_at": time.monotonic(),
    }


def wait_for_job(
    submitted: dict,
    *,
    cleanup_input: bool = True,
) -> dict:
    job = submitted["job"]

    while True:
        status = normalize_status(job)

        print(f"{job.job_id}: {status}")

        if status == "COMPLETED":
            result = job.output()

            if cleanup_input:
                delete_input(submitted["input_key"])

            return result

        if status in {
            "FAILED",
            "CANCELLED",
            "TIMED_OUT",
        }:
            raise RuntimeError(f"Job {job.job_id} ended with {status}")

        if time.monotonic() - submitted["submitted_at"] > RUNPOD_JOB_TIMEOUT:
            raise TimeoutError(f"Client timeout waiting for {job.job_id}")

        time.sleep(RUNPOD_POLL_INTERVAL)


# ---------------------------------------------------------------------
# Parallel batch
# ---------------------------------------------------------------------


def run_batch(
    documents: list[dict],
    *,
    cleanup_inputs: bool = True,
) -> list[dict]:

    pending = list(documents)
    active = {}
    completed = []

    while pending or active:
        state = get_capacity()

        capacity_per_worker = state["capacity_per_worker"]

        target_in_flight = max(
            capacity_per_worker,
            state["warm_capacity"] + RUNPOD_AUTOSCALE_QUEUE_BUFFER,
        )

        while pending and len(active) < target_in_flight:
            document = pending.pop(0)

            submitted = submit_document(
                document["file"],
                tier=document.get(
                    "tier",
                    "standard",
                ),
                pages=document.get(
                    "pages",
                    "all",
                ),
            )

            job = submitted["job"]

            active[job.job_id] = submitted

            print(f"SUBMIT {job.job_id} {document['file']}")

        print(
            "\n"
            f"workers "
            f"idle={state['idle_workers']} "
            f"running={state['running_workers']} "
            f"ready={state['ready_workers']} "
            f"initializing={state['initializing_workers']}\n"
            f"capacity/worker="
            f"{state['capacity_per_worker']} "
            f"warm_capacity="
            f"{state['warm_capacity']} "
            f"potential_capacity="
            f"{state['potential_capacity']}\n"
            f"queue={state['in_queue']} "
            f"processing={state['in_progress']} "
            f"client_active={len(active)} "
            f"pending={len(pending)}"
        )

        for job_id, submitted in list(active.items()):
            job = submitted["job"]

            try:
                status = normalize_status(job)
            except Exception as exc:
                print(f"STATUS ERROR {job_id}: {exc}")
                continue

            if status == "COMPLETED":
                result = job.output()

                if cleanup_inputs:
                    delete_input(submitted["input_key"])

                completed.append(
                    {
                        "job_id": job_id,
                        "source": submitted["source"],
                        "status": status,
                        "result": result,
                    }
                )

                del active[job_id]

                print(f"DONE {job_id} s3://{result['bucket']}/{result['key']}")

            elif status in {
                "FAILED",
                "CANCELLED",
                "TIMED_OUT",
            }:
                completed.append(
                    {
                        "job_id": job_id,
                        "source": submitted["source"],
                        "status": status,
                    }
                )

                del active[job_id]

                print(f"{status} {job_id}")

            elif time.monotonic() - submitted["submitted_at"] > RUNPOD_JOB_TIMEOUT:
                completed.append(
                    {
                        "job_id": job_id,
                        "source": submitted["source"],
                        "status": "CLIENT_TIMEOUT",
                    }
                )

                del active[job_id]

                print(f"CLIENT_TIMEOUT {job_id}")

        if pending or active:
            time.sleep(RUNPOD_POLL_INTERVAL)

    return completed


# ---------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------

if __name__ == "__main__":
    documents = [
        {
            "file": "pdfs/documento1.pdf",
            "tier": "standard",
        },
        {
            "file": "pdfs/documento2.pdf",
            "tier": "standard",
        },
        {
            "file": "pdfs/documento3.pdf",
            "tier": "standard",
        },
    ]

    results = run_batch(documents)

    for result in results:
        print(result)
