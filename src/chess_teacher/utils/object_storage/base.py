from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager

from chess_teacher.utils.exception_utils import FileError
from chess_teacher.utils.files.file_utils import TextStreamSource
from chess_teacher.utils.object_storage.keys import (
    key_basename,
    relative_key_under,
    sibling_temp_key,
    unique_key_variant,
)


class ObjectStorage(ABC):
    """Abstract object storage backend keyed by POSIX-style paths."""

    @staticmethod
    def resolve_key(*parts: str) -> str:
        """Join key parts with ``/``, stripping leading and trailing slashes on each part."""
        normalized = [part.strip("/") for part in parts if part]
        return "/".join(normalized)

    key_basename = staticmethod(key_basename)
    relative_key_under = staticmethod(relative_key_under)
    unique_key_variant = staticmethod(unique_key_variant)
    sibling_temp_key = staticmethod(sibling_temp_key)

    @abstractmethod
    def open_text(
        self, key: str, *, encoding: str = "utf-8-sig"
    ) -> AbstractContextManager[TextStreamSource]:
        """Open a text object for reading."""
        pass

    @abstractmethod
    def read_bytes(self, key: str) -> bytes | None:
        """Read an object's raw bytes, or ``None`` if the object does not exist."""
        pass

    @abstractmethod
    def write_bytes(self, key: str, data: bytes, *, overwrite: bool = False) -> None:
        """Write raw bytes to an object."""
        pass

    @abstractmethod
    def write_text_atomic(
        self,
        key: str,
        text: str,
        *,
        encoding: str = "utf-8",
        overwrite: bool = False,
    ) -> None:
        """Write text atomically via a temporary object and move."""
        pass

    @abstractmethod
    def list_keys(
        self,
        prefix: str = "",
        *,
        recursive: bool = True,
        suffix: str | None = None,
        glob_pattern: str | None = None,
    ) -> list[str]:
        """List object keys under ``prefix``, relative to the storage root."""
        pass

    @abstractmethod
    def move(self, source_key: str, dest_key: str, *, overwrite: bool = False) -> None:
        """Move an object from ``source_key`` to ``dest_key``."""
        pass

    def move_verified(self, source_key: str, dest_key: str, *, overwrite: bool = False) -> None:
        """Move an object and ensure the source key no longer exists afterward."""
        self.move(source_key, dest_key, overwrite=overwrite)
        if self.read_bytes(dest_key) is None:
            raise FileError(f"Destination missing after move: {dest_key!r}")
        if self.read_bytes(source_key) is not None:
            self.delete(source_key, missing_ok=False)
        if self.read_bytes(source_key) is not None:
            raise FileError(f"Source object still present after move: {source_key!r}")

    @abstractmethod
    def delete(self, key: str, *, missing_ok: bool = True) -> None:
        """Delete a single object."""
        pass

    @abstractmethod
    def delete_keys(self, keys: list[str], *, missing_ok: bool = True) -> None:
        """Delete multiple objects."""
        pass

    def presigned_get_url(self, key: str, *, expires_in: int = 3600) -> str | None:
        """Return a time-limited HTTPS URL for reading an object, or ``None`` if unsupported."""
        _ = key, expires_in
        return None
