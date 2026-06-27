from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from chess_teacher.utils.exception_utils import FileError
from chess_teacher.utils.object_storage.s3 import S3ObjectStorage


@pytest.fixture
def s3_storage() -> S3ObjectStorage:
    store = S3ObjectStorage(
        bucket="test-bucket",
        key_prefix="storage/raw",
        endpoint_url="https://example.com",
        access_key="key",
        secret_key="secret",
    )
    store._client = MagicMock()
    return store


def test_delete_keys_raises_on_partial_failure(s3_storage: S3ObjectStorage) -> None:
    s3_storage._client.delete_objects.return_value = {
        "Errors": [
            {
                "Key": "storage/raw/ingested/a.jsonl",
                "Code": "AccessDenied",
                "Message": "Forbidden",
            }
        ]
    }

    with pytest.raises(FileError, match="AccessDenied"):
        s3_storage.delete_keys(["ingested/a.jsonl"], missing_ok=True)


def test_delete_keys_ignores_missing_objects(s3_storage: S3ObjectStorage) -> None:
    s3_storage._client.delete_objects.return_value = {
        "Errors": [
            {
                "Key": "storage/raw/ingested/missing.jsonl",
                "Code": "NoSuchKey",
                "Message": "Not found",
            }
        ]
    }

    s3_storage.delete_keys(["ingested/missing.jsonl"], missing_ok=True)


def test_delete_keys_raises_on_client_error_when_not_missing_ok(
    s3_storage: S3ObjectStorage,
) -> None:
    s3_storage._client.delete_objects.side_effect = ClientError(
        {"Error": {"Code": "500", "Message": "Server error"}},
        "DeleteObjects",
    )

    with pytest.raises(FileError, match="Could not delete objects"):
        s3_storage.delete_keys(["ingested/a.jsonl"], missing_ok=False)
