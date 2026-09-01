"""Resolve Keras ``.keras`` weight files from URIs without importing MLflow when possible."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.object_storage.base import ObjectStorage
from chess_teacher.utils.object_storage.factory import (
    build_s3_storage_settings,
    get_raw_storage,
)

logger = get_logger()

_CACHE_ENV = "CHESS_TEACHER_MODEL_CACHE_DIR"


def parse_s3_uri(uri: str) -> tuple[str, str] | None:
    """Return ``(bucket, key)`` for ``s3://bucket/key`` URIs."""
    if not uri.startswith("s3://"):
        return None
    parsed = urlparse(uri)
    bucket = parsed.netloc.strip()
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        return None
    return bucket, key


def s3_key_to_storage_relative(full_key: str, *, key_prefix: str) -> str:
    """Map a bucket object key to ``ObjectStorage`` relative key."""
    prefix = key_prefix.strip("/")
    if prefix and full_key.startswith(prefix + "/"):
        return full_key[len(prefix) + 1 :]
    if prefix and full_key == prefix:
        return ""
    return full_key


def _model_cache_dir() -> Path:
    raw = os.environ.get(_CACHE_ENV, "").strip()
    if raw:
        path = Path(raw)
    else:
        path = Path(tempfile.gettempdir()) / "chess_teacher_model_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_path_for_uri(uri: str, *, filename: str) -> Path:
    digest = hashlib.sha256(uri.encode("utf-8")).hexdigest()[:16]
    return _model_cache_dir() / digest / filename


def _write_cached(uri: str, data: bytes, *, filename: str) -> Path:
    dest = _cache_path_for_uri(uri, filename=filename)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest


def _resolve_local_path(model_uri: str) -> Path | None:
    if model_uri.startswith("file:"):
        path = Path(model_uri.removeprefix("file:"))
        return path if path.is_file() else None
    candidate = Path(model_uri)
    return candidate if candidate.is_file() else None


def _resolve_s3_keras_weights(
    model_uri: str,
    *,
    storage: ObjectStorage | None = None,
) -> Path | None:
    parsed = parse_s3_uri(model_uri)
    if parsed is None:
        return None
    bucket, full_key = parsed
    cfg = build_s3_storage_settings()
    if bucket != cfg.bucket:
        logger.debug(
            "S3 model URI bucket %r != configured %r; skip direct storage fetch",
            bucket,
            cfg.bucket,
        )
        return None

    store = storage if storage is not None else get_raw_storage()
    relative = s3_key_to_storage_relative(full_key, key_prefix=cfg.key_prefix)

    if full_key.endswith(".keras"):
        cached = _cache_path_for_uri(model_uri, filename=Path(full_key).name)
        if cached.is_file():
            return cached
        data = store.read_bytes(relative)
        if data is None:
            return None
        return _write_cached(model_uri, data, filename=Path(full_key).name)

    # Artifact directory: find first ``.keras`` object under prefix.
    prefix = relative.rstrip("/") + "/" if relative else ""
    keys = store.list_keys(prefix, recursive=True, glob_pattern=r".*\.keras$")
    if not keys:
        return None
    keras_key = sorted(keys)[0]
    filename = Path(keras_key).name
    cached = _cache_path_for_uri(model_uri, filename=filename)
    if cached.is_file():
        return cached
    data = store.read_bytes(keras_key)
    if data is None:
        return None
    return _write_cached(model_uri, data, filename=filename)


def resolve_keras_weights_path(
    model_uri: str,
    *,
    storage: ObjectStorage | None = None,
) -> Path | None:
    """Resolve a local ``.keras`` path from file / S3 URIs without MLflow."""
    if not model_uri:
        return None
    local = _resolve_local_path(model_uri)
    if local is not None:
        return local
    if model_uri.startswith("s3://"):
        return _resolve_s3_keras_weights(model_uri, storage=storage)
    return None


def require_keras_weights_path(
    model_uri: str,
    *,
    storage: ObjectStorage | None = None,
) -> Path:
    """Like :func:`resolve_keras_weights_path` but raises when missing."""
    path = resolve_keras_weights_path(model_uri, storage=storage)
    if path is None or not path.is_file():
        raise FileNotFoundError(f"Could not resolve Keras weights from uri={model_uri!r}")
    return path
