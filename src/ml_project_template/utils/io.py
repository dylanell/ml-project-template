"""S3-compatible I/O utilities."""

import os


def get_storage_options() -> dict:
    """Return S3 storage options for pandas.

    Returns a dict suitable for passing to pandas read_csv/to_csv as
    storage_options. Returns empty dict if S3_ENDPOINT_URL is not set,
    so local paths work transparently.
    """
    endpoint = os.environ.get("S3_ENDPOINT_URL")
    if not endpoint:
        return {}
    return {
        "endpoint_url": endpoint,
        "key": os.environ["AWS_ACCESS_KEY_ID"],
        "secret": os.environ["AWS_SECRET_ACCESS_KEY"],
    }


def get_s3_filesystem():
    """Return a configured s3fs.S3FileSystem instance."""
    import s3fs
    return s3fs.S3FileSystem(
        endpoint_url=os.environ["S3_ENDPOINT_URL"],
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
