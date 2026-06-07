"""Lightweight read/write/delete probe for raw object storage."""

from __future__ import annotations

from uuid import uuid4

from chess_teacher.utils.exception_utils import FileError
from chess_teacher.utils.object_storage.base import ObjectStorage
from chess_teacher.utils.object_storage.factory import get_raw_storage

_HEALTHCHECK_PREFIX = "_healthcheck"
_PROBE_PAYLOAD = b"chess_teacher_storage_ok"


def check_raw_storage_health(storage: ObjectStorage | None = None) -> None:
    """Verify raw storage can write, read, list, and delete a small probe object.

    Raises:
        FileError: if any operation fails or round-trip content differs.
    """
    store = storage if storage is not None else get_raw_storage()
    key = ObjectStorage.resolve_key(_HEALTHCHECK_PREFIX, f"{uuid4().hex}.txt")

    store.write_bytes(key, _PROBE_PAYLOAD, overwrite=True)
    try:
        read_back = store.read_bytes(key)
        if read_back != _PROBE_PAYLOAD:
            raise FileError(
                f"Storage health check failed for {key!r}: "
                f"expected {_PROBE_PAYLOAD!r}, got {read_back!r}"
            )

        listed = store.list_keys(_HEALTHCHECK_PREFIX, recursive=True)
        if key not in listed:
            raise FileError(
                f"Storage health check failed: probe key {key!r} missing from list_keys result"
            )
    finally:
        store.delete(key, missing_ok=True)
