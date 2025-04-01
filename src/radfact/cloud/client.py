"""Stripped down cloud clients with only single file download/upload"""

from pathlib import Path
from typing import ClassVar

import boto3
from google.cloud import storage

from .types import GCSPath, S3Path


def get_bucket_and_key_s3(s3_path: S3Path) -> tuple[str, str]:
    """Extracts the bucket and key from an S3 path.

    Args:
        s3_path: The S3 path to extract the bucket and key from.

    Returns:
        A tuple containing the bucket and key.
    """
    validate_s3_path(s3_path)
    return str(s3_path.parts[1]), str(Path(*s3_path.parts[2:]))


def validate_s3_path(s3_path: S3Path) -> None:
    if s3_path.parts[0] != "s3:":
        raise ValueError(f"Invalid S3 path: {s3_path}")


def get_bucket_and_key_gcs(gcs_path: GCSPath) -> tuple[str, str]:
    """Extracts the bucket and key from an GCS path.

    Args:
        gcs_path: The GCS path to extract the bucket and key from.

    Returns:
        A tuple containing the bucket and key.
    """
    validate_gcs_path(gcs_path)
    return str(gcs_path.parts[1]), str(Path(*gcs_path.parts[2:]))


def validate_gcs_path(gcs_path: GCSPath) -> None:
    if gcs_path.parts[0] != "gs:":
        raise ValueError(f"Invalid GCS path: {gcs_path}")


class S3Client:

    _boto3_client: ClassVar[boto3.client] = None

    @classmethod
    def download_file(cls, s3_filepath: S3Path, filepath: Path) -> None:
        cls._ensure_boto3_client()
        bucket, key = get_bucket_and_key_s3(s3_filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        cls._boto3_client.download_file(Bucket=bucket, Key=key, Filename=str(filepath))

    @classmethod
    def upload_file(cls, filepath: Path, s3_filepath: S3Path) -> None:
        cls._ensure_boto3_client()
        bucket, key = get_bucket_and_key_s3(s3_filepath)
        cls._boto3_client.upload_file(Bucket=bucket, Key=key, Filename=str(filepath))


class GCSClient:

    _gcs_client: ClassVar[storage.Client] = None

    @classmethod
    def download_file(
        cls,
        gcs_filepath: GCSPath,
        destination_file_path: Path,
        override_existing: bool = False,
    ) -> None:
        if not override_existing and destination_file_path.exists():
            return
        cls._ensure_gcs_client()
        bucket_name, blob_name = get_bucket_and_key_gcs(gcs_filepath)
        bucket = cls._gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        destination_file_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(destination_file_path)

    @classmethod
    def upload_file(cls, source_file_path: Path, gcs_filepath: GCSPath) -> None:
        cls._ensure_gcs_client()
        bucket_name, blob_name = get_bucket_and_key_gcs(gcs_filepath)
        bucket = cls._gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(source_file_path)
