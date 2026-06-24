"""Image helpers for raw object storage (keys, MIME types, data URIs)."""

from __future__ import annotations

import base64
import functools
from pathlib import Path

from chess_teacher.utils.object_storage.base import ObjectStorage
from chess_teacher.utils.object_storage.factory import get_raw_storage

ASSET_IMAGES_PREFIX = "assets/images"


def asset_image_key(filename: str) -> str:
    """Storage key for a file under ``assets/images/``."""
    return ObjectStorage.resolve_key(ASSET_IMAGES_PREFIX, filename)


def read_asset_image(filename: str, *, storage: ObjectStorage | None = None) -> bytes | None:
    """Read an image from ``assets/images/{filename}``."""
    return read_raw_object(asset_image_key(filename), storage=storage)


def read_raw_object(key: str, *, storage: ObjectStorage | None = None) -> bytes | None:
    """Read bytes for any key in raw object storage."""
    store = storage if storage is not None else get_raw_storage()
    return store.read_bytes(key)


def mime_type_for_key(key: str) -> str:
    """Guess an image MIME type from a storage key suffix."""
    suffix = Path(key).suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".gif":
        return "image/gif"
    if suffix == ".svg":
        return "image/svg+xml"
    return "application/octet-stream"


def bytes_to_data_uri(data: bytes, mime: str) -> str:
    """Encode raw bytes as a ``data:`` URI."""
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _build_storage_image_data_uri(key: str, storage: ObjectStorage) -> str | None:
    data = storage.read_bytes(key)
    if data is None:
        return None
    return bytes_to_data_uri(data, mime_type_for_key(key))


def storage_image_data_uri(key: str, *, storage: ObjectStorage | None = None) -> str | None:
    """Return a cached ``data:`` URI for an immutable storage object (e.g. bundled logos)."""
    if storage is not None:
        return _build_storage_image_data_uri(key, storage)
    return _cached_storage_image_data_uri(key)


@functools.lru_cache(maxsize=128)
def _cached_storage_image_data_uri(key: str) -> str | None:
    return _build_storage_image_data_uri(key, get_raw_storage())


def clear_storage_image_data_uri_cache() -> None:
    """Clear the process-wide data-URI cache (for tests or asset updates)."""
    _cached_storage_image_data_uri.cache_clear()
