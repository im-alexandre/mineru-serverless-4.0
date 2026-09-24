import os

import boto3
import runpod
from botocore.client import Config

runpod.api_key = os.getenv("RUNPOD_API_KEY")

endpoint = runpod.Endpoint("3aw008zvs0ilqd")

# Configure the S3 client to point to your SeaweedFS S3 endpoint
s3_client = boto3.client(
    "s3",
    endpoint_url="https://storage.baseia.ai",  # Replace with your SeaweedFS S3 endpoint
    aws_access_key_id="baseia",  # Replace with your SeaweedFS access key
    aws_secret_access_key="baseia",  # Replace with your SeaweedFS secret key
    config=Config(signature_version="s3v4"),  # Recommended for S3 compatibility
)

# Generate a presigned URL to get/download an object
presigned_url = s3_client.generate_presigned_url(
    ClientMethod="get_object",
    Params={
        "Bucket": "baseia-marinha",  # Replace with your bucket name
        "Key": "documentos/SGM-303-Rev7.pdf",  # Replace with your object key/path
    },
    ExpiresIn=3600,  # URL validity in seconds (1 hour)
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
