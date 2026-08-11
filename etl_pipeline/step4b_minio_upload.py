"""
Step 4b -- Upload raw Reddit CSV to MinIO
Uploads S_t_raw_AAPL_reddit.csv to the raw-reddit bucket.
"""

import boto3 # Amazon's python library for talking to S3
from botocore.client import Config
import os

MINIO_URL   = "http://localhost:9000" # we tell boto3 to speak to MinIO in localhost:9000 instead to S3
ACCESS_KEY  = "minioadmin"
SECRET_KEY  = "minioadmin"
BUCKET_NAME = "raw-reddit"
LOCAL_FILE  = os.path.join("data","S_t_raw_AAPL_reddit.csv")
OBJECT_NAME = "AAPL/S_t_raw_AAPL_reddit.csv"

def main():
    client = boto3.client(
        "s3",
        endpoint_url = MINIO_URL,
        aws_access_key_id = ACCESS_KEY,
        aws_secret_access_key = SECRET_KEY,
        config = Config(signature_version = "s3v4"),
    )

    print(f"Uploading {LOCAL_FILE} to {BUCKET_NAME}/{OBJECT_NAME}...")
    client.upload_file(LOCAL_FILE, BUCKET_NAME, OBJECT_NAME)
    print("Upload complete")


if __name__ == "__main__":
    main()