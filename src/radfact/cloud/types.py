from pathlib import PurePosixPath


class GCSPath(PurePosixPath):
    def __str__(self) -> str:
        return f"gs://{'/'.join(self.parts[1:])}"


class S3Path(PurePosixPath):
    def __str__(self) -> str:
        return f"s3://{'/'.join(self.parts[1:])}"
