"""Object storage for raw data (S3-compatible buckets; legacy local backend for tests).

Import concrete modules (``base``, ``factory``, ``s3``, …) directly.
Package-level names below load lazily so ``import …object_storage.base`` does not
pull factory/logging/shipping at import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chess_teacher.utils.object_storage.base import ObjectStorage
    from chess_teacher.utils.object_storage.factory import S3StorageSettings
    from chess_teacher.utils.object_storage.filesystem import FilesystemObjectStorage
    from chess_teacher.utils.object_storage.s3 import S3ObjectStorage

__all__ = [
    "FilesystemObjectStorage",
    "ObjectStorage",
    "S3ObjectStorage",
    "S3StorageSettings",
    "asset_image_key",
    "build_s3_storage_settings",
    "bytes_to_data_uri",
    "check_raw_storage_health",
    "clear_storage_image_data_uri_cache",
    "get_local_log_dir",
    "get_log_storage_key",
    "get_raw_storage",
    "key_basename",
    "mime_type_for_key",
    "read_asset_image",
    "read_raw_object",
    "relative_key_under",
    "reset_raw_storage",
    "s3_url_string",
    "sibling_temp_key",
    "storage_image_data_uri",
    "unique_key_variant",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "ObjectStorage": (".base", "ObjectStorage"),
    "FilesystemObjectStorage": (".filesystem", "FilesystemObjectStorage"),
    "S3ObjectStorage": (".s3", "S3ObjectStorage"),
    "S3StorageSettings": (".factory", "S3StorageSettings"),
    "build_s3_storage_settings": (".factory", "build_s3_storage_settings"),
    "get_local_log_dir": (".factory", "get_local_log_dir"),
    "get_log_storage_key": (".factory", "get_log_storage_key"),
    "get_raw_storage": (".factory", "get_raw_storage"),
    "reset_raw_storage": (".factory", "reset_raw_storage"),
    "s3_url_string": (".factory", "s3_url_string"),
    "check_raw_storage_health": (".health", "check_raw_storage_health"),
    "asset_image_key": (".images", "asset_image_key"),
    "bytes_to_data_uri": (".images", "bytes_to_data_uri"),
    "clear_storage_image_data_uri_cache": (".images", "clear_storage_image_data_uri_cache"),
    "mime_type_for_key": (".images", "mime_type_for_key"),
    "read_asset_image": (".images", "read_asset_image"),
    "read_raw_object": (".images", "read_raw_object"),
    "storage_image_data_uri": (".images", "storage_image_data_uri"),
    "key_basename": (".keys", "key_basename"),
    "relative_key_under": (".keys", "relative_key_under"),
    "sibling_temp_key": (".keys", "sibling_temp_key"),
    "unique_key_variant": (".keys", "unique_key_variant"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_name, attr = _LAZY_ATTRS[name]
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr)
    globals()[name] = value
    return value
