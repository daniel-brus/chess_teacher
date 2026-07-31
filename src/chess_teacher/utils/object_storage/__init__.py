"""Object storage for raw data (S3-compatible buckets; legacy local backend for tests)."""

from chess_teacher.utils.object_storage.base import ObjectStorage
from chess_teacher.utils.object_storage.factory import (
    S3StorageSettings,
    build_s3_storage_settings,
    get_local_log_dir,
    get_log_storage_key,
    get_raw_storage,
    reset_raw_storage,
    s3_url_string,
)
from chess_teacher.utils.object_storage.filesystem import FilesystemObjectStorage
from chess_teacher.utils.object_storage.health import check_raw_storage_health
from chess_teacher.utils.object_storage.images import (
    asset_image_key,
    bytes_to_data_uri,
    clear_storage_image_data_uri_cache,
    mime_type_for_key,
    read_asset_image,
    read_raw_object,
    storage_image_data_uri,
)
from chess_teacher.utils.object_storage.keys import (
    key_basename,
    relative_key_under,
    sibling_temp_key,
    unique_key_variant,
)
from chess_teacher.utils.object_storage.s3 import S3ObjectStorage

from . import factory

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
    "factory",
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
