import os
import time

import requests
import runpod

RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]

#
# Tem que ser igual à configuração do endpoint.
#
GPUS_PER_WORKER = int(
    os.environ.get(
        "RUNPOD_GPUS_PER_WORKER",
        "1",
    )
)

CONCURRENCY_PER_GPU = int(
    os.environ.get(
        "MINERU_CONCURRENCY_PER_GPU",
        "3",
    )
)

CAPACITY_PER_WORKER = GPUS_PER_WORKER * CONCURRENCY_PER_GPU

POLL_INTERVAL = 2

#
# Um pequeno backlog é importante para o
# autoscaler perceber demanda.
#
AUTOSCALE_QUEUE_BUFFER = CAPACITY_PER_WORKER


endpoint = runpod.Endpoint(
    ENDPOINT_ID,
    api_key=RUNPOD_API_KEY,
)


def get_endpoint_health() -> dict:

    response = requests.get(
        (f"https://api.runpod.ai/v2/{ENDPOINT_ID}/health"),
        headers={
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
        },
        timeout=10,
    )

    response.raise_for_status()

    return response.json()


def get_capacity() -> dict:

    health = get_endpoint_health()

    workers = health.get(
        "workers",
        {},
    )

    jobs = health.get(
        "jobs",
        {},
    )

    idle_workers = int(workers.get("idle", 0))

    running_workers = int(workers.get("running", 0))

    ready_workers = int(workers.get("ready", 0))

    initializing_workers = int(workers.get("initializing", 0))

    #
    # Avoid double counting if RunPod reports
    # ready as an aggregate.
    #
    warm_workers = max(
        ready_workers,
        idle_workers + running_workers,
    )

    warm_capacity = warm_workers * CAPACITY_PER_WORKER

    potential_capacity = (warm_workers + initializing_workers) * CAPACITY_PER_WORKER

    return {
        "idle_workers": idle_workers,
        "running_workers": running_workers,
        "ready_workers": ready_workers,
        "initializing_workers": initializing_workers,
        "capacity_per_worker": CAPACITY_PER_WORKER,
        "warm_capacity": warm_capacity,
        "potential_capacity": potential_capacity,
        "queue": int(
            jobs.get(
                "inQueue",
                0,
            )
        ),
        "in_progress": int(
            jobs.get(
                "inProgress",
                0,
            )
        ),
    }


def job_status(job):

    result = job.status()

    if isinstance(result, dict):
        return result.get("status", "").upper()

    return str(result).upper()


def run_batch(
    documents: list[dict],
):

    pending = list(documents)

    active = {}

    completed = []

    while pending or active:
        state = get_capacity()

        #
        # Keep enough jobs submitted to:
        #
        # 1. saturate warm workers
        # 2. leave enough queue pressure
        #    for another worker to spawn
        #
        target_in_flight = max(
            CAPACITY_PER_WORKER,
            state["warm_capacity"] + AUTOSCALE_QUEUE_BUFFER,
        )

        while pending and len(active) < target_in_flight:
            document = pending.pop(0)

            job = endpoint.run(document)

            active[job.job_id] = {
                "job": job,
                "input": document,
                "started": time.monotonic(),
            }

            print(f"SUBMIT {job.job_id} {document['url']}")

        print(
            "\n"
            f"workers: "
            f"idle={state['idle_workers']} "
            f"running={state['running_workers']} "
            f"initializing="
            f"{state['initializing_workers']} "
            f"ready={state['ready_workers']}\n"
            f"capacity/worker="
            f"{state['capacity_per_worker']} "
            f"warm_capacity="
            f"{state['warm_capacity']} "
            f"potential_capacity="
            f"{state['potential_capacity']}\n"
            f"runpod_queue="
            f"{state['queue']} "
            f"runpod_processing="
            f"{state['in_progress']} "
            f"client_active="
            f"{len(active)} "
            f"pending="
            f"{len(pending)}"
        )

        for job_id, item in list(active.items()):
            job = item["job"]

            try:
                status = job_status(job)
            except Exception as exc:
                print(f"STATUS ERROR {job_id}: {exc}")
                continue

            if status == "COMPLETED":
                result = job.output()

                completed.append(
                    {
                        "job_id": job_id,
                        "status": status,
                        "input": item["input"],
                        "result": result,
                    }
                )

                del active[job_id]

                print(f"DONE {job_id} {result.get('key')}")

            elif status in {
                "FAILED",
                "CANCELLED",
                "TIMED_OUT",
            }:
                completed.append(
                    {
                        "job_id": job_id,
                        "status": status,
                        "input": item["input"],
                    }
                )

                del active[job_id]

                print(f"{status} {job_id}")

        if pending or active:
            time.sleep(POLL_INTERVAL)

    return completed


if __name__ == "__main__":
    documents = [
        {
            "url": "https://origem/documento1.pdf",
            "tier": "standard",
            "destination": {
                "bucket": "baseia",
                "key": ("collections/123/documents/abc/mineru-output.tar.gz"),
            },
        },
        {
            "url": "https://origem/documento2.pdf",
            "tier": "standard",
            "destination": {
                "bucket": "baseia",
                "key": ("collections/123/documents/def/mineru-output.tar.gz"),
            },
        },
    ]

    results = run_batch(documents)

    print(results)
