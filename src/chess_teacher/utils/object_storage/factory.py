from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from chess_teacher.utils.env_utils import get_env_variable
from chess_teacher.utils.exception_utils import ConfigError
from chess_teacher.utils.logging.logger import EnhancedLogger
from chess_teacher.utils.object_storage.base import ObjectStorage

_raw_storage: ObjectStorage | None = None
_logger: EnhancedLogger | None = None


def _get_logger() -> EnhancedLogger:
    global _logger
    if _logger is None:
        from chess_teacher.utils.logging.config import get_logger

        _logger = get_logger()
    return _logger


@dataclass(frozen=True)
class S3StorageSettings:
    """S3-compatible connection settings (parallel to SQLAlchemy Postgres ``URL``)."""

    bucket: str
    key_prefix: str
    endpoint_url: str
    access_key: str
    secret_key: str

    def object_uri(self, *parts: str) -> str:
        """Build ``s3://bucket/prefix/...parts`` (empty parts skipped)."""
        segments = [self.key_prefix.strip("/")]
        segments.extend(p.strip("/") for p in parts if p and p.strip("/"))
        path = "/".join(seg for seg in segments if seg)
        return f"s3://{self.bucket}/{path}" if path else f"s3://{self.bucket}"


def build_s3_storage_settings(
    *,
    bucket: str = "",
    key_prefix: str = "",
    endpoint_url: str = "",
    access_key: str = "",
    secret_key: str = "",
) -> S3StorageSettings:
    """Build S3 settings from args, falling back to ``S3_*`` / ``STORAGE_ROOT`` env.

    Parallel to ``build_postgres_url`` in ``db.engine``.
    """
    try:
        return S3StorageSettings(
            bucket=bucket or get_env_variable("S3_BUCKET"),
            key_prefix=key_prefix or get_env_variable("STORAGE_ROOT"),
            endpoint_url=endpoint_url or get_env_variable("S3_ENDPOINT_URL"),
            access_key=access_key or get_env_variable("S3_ACCESS_KEY_ID"),
            secret_key=secret_key or get_env_variable("S3_SECRET_ACCESS_KEY"),
        )
    except Exception as e:
        _get_logger().log_and_raise(
            ConfigError(f"Error occurred while fetching S3 storage credentials: {e}")
        )


def s3_url_string(
    *parts: str,
    bucket: str = "",
    key_prefix: str = "",
    endpoint_url: str = "",
    access_key: str = "",
    secret_key: str = "",
) -> str:
    """Render an ``s3://`` URI under ``STORAGE_ROOT`` (parallel to ``postgres_url_string``).

    Location only — credentials are not embedded (unlike Postgres URLs).
    """
    return build_s3_storage_settings(
        bucket=bucket,
        key_prefix=key_prefix,
        endpoint_url=endpoint_url,
        access_key=access_key,
        secret_key=secret_key,
    ).object_uri(*parts)


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
    from chess_teacher.utils.object_storage.s3 import S3ObjectStorage

    cfg = build_s3_storage_settings()
    _get_logger().info(
        "Object storage backend=s3 bucket=%s endpoint=%s key_prefix=%s",
        cfg.bucket,
        cfg.endpoint_url,
        cfg.key_prefix,
    )
    return S3ObjectStorage(
        bucket=cfg.bucket,
        key_prefix=cfg.key_prefix,
        endpoint_url=cfg.endpoint_url,
        access_key=cfg.access_key,
        secret_key=cfg.secret_key,
    )
