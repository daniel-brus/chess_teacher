from __future__ import annotations

from pathlib import Path

from chess_teacher.utils.env_utils import get_env_variable
from chess_teacher.utils.exception_utils import ConfigError
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.object_storage.base import ObjectStorage
from chess_teacher.utils.object_storage.filesystem import FilesystemObjectStorage

logger = get_logger()

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
    """Return the local filesystem directory for Python log buffering."""
    from chess_teacher.utils.logging.buffer import get_log_buffer_dir

    return get_log_buffer_dir()


def get_log_storage_key(relative_under_buffer: str) -> str:
    """Build an object storage key for a path relative to the log buffer root."""
    return ObjectStorage.resolve_key("logs/python/buffer", relative_under_buffer)


def _create_raw_storage() -> ObjectStorage:
    backend = get_env_variable("STORAGE_BACKEND")
    match backend:
        case "filesystem":
            root = Path(get_env_variable("STORAGE_ROOT"))
            logger.info("Object storage backend=filesystem root=%s", root)
            return FilesystemObjectStorage(root)
        case "s3":
            from chess_teacher.utils.object_storage.s3 import S3ObjectStorage

            bucket = get_env_variable("S3_BUCKET")
            key_prefix = get_env_variable("STORAGE_ROOT")
            endpoint_url = get_env_variable("S3_ENDPOINT_URL")
            logger.info(
                "Object storage backend=s3 bucket=%s endpoint=%s key_prefix=%s",
                bucket,
                endpoint_url,
                key_prefix,
            )
            return S3ObjectStorage(
                bucket=bucket,
                key_prefix=key_prefix,
                endpoint_url=endpoint_url,
                access_key=get_env_variable("S3_ACCESS_KEY_ID"),
                secret_key=get_env_variable("S3_SECRET_ACCESS_KEY"),
            )
        case _:
            raise ConfigError(f"Unsupported STORAGE_BACKEND: {backend!r}")
