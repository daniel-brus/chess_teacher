from __future__ import annotations

from pathlib import Path

from chess_teacher.utils.env_utils import get_env_variable
from chess_teacher.utils.exception_utils import ConfigError
from chess_teacher.utils.object_storage.base import ObjectStorage
from chess_teacher.utils.object_storage.filesystem import FilesystemObjectStorage

_raw_storage: ObjectStorage | None = None


def get_raw_storage() -> ObjectStorage:
    """Return the lazily initialized raw object storage singleton."""
    global _raw_storage
    if _raw_storage is None:
        _raw_storage = _create_raw_storage()
    return _raw_storage


def reset_raw_storage() -> None:
    """Clear the raw storage singleton (for tests)."""
    global _raw_storage
    _raw_storage = None


def get_local_log_dir() -> Path:
    """Return the local filesystem directory for Python logs."""
    return Path(get_env_variable("RAW_DIR")) / "logs" / "python"


def _create_raw_storage() -> ObjectStorage:
    backend = get_env_variable("STORAGE_BACKEND")
    match backend:
        case "filesystem":
            return FilesystemObjectStorage(Path(get_env_variable("RAW_DIR")))
        case "s3":
            from chess_teacher.utils.object_storage.s3 import S3ObjectStorage

            return S3ObjectStorage(
                bucket=get_env_variable("S3_BUCKET"),
                key_prefix=get_env_variable("RAW_DIR"),
                endpoint_url=get_env_variable("S3_ENDPOINT_URL"),
                access_key=get_env_variable("S3_ACCESS_KEY_ID"),
                secret_key=get_env_variable("S3_SECRET_ACCESS_KEY"),
            )
        case _:
            raise ConfigError(f"Unsupported STORAGE_BACKEND: {backend!r}")
