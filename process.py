import os

import boto3
import runpod
from botocore.client import Config
from dotenv import load_dotenv

load_dotenv()

runpod.api_key = os.environ["RUNPOD_API_KEY"]

endpoint = runpod.Endpoint(os.environ["RUNPOD_ENDPOINT_ID"])

s3_client = boto3.client(
    "s3",
    endpoint_url=os.environ["S3_ENDPOINT_URL"],
    aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
    config=Config(signature_version="s3v4"),
)

presigned_url = s3_client.generate_presigned_url(
    ClientMethod="get_object",
    Params={
        "Bucket": os.environ["S3_BUCKET"],
        "Key": os.environ["EXAMPLE_S3_KEY"],
    },
    ExpiresIn=3600,
)

result = endpoint.run_sync(
    {
        "url": presigned_url,
        "tier": "standard",
        "pages": "all",
    },
    timeout=1800,
)

print(result)
