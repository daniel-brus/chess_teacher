from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from chess_teacher.utils.exception_utils import FileError
from chess_teacher.utils.file_utils import (
    discover_files,
    ensure_destination_parent,
    move_file,
    remove_file,
    validate_existing_file,
)
from chess_teacher.utils.files.text_stream_source import TextStreamSource
from chess_teacher.utils.logging_utils import EnhancedLogger, get_logger
from chess_teacher.utils.object_storage.base import ObjectStorage


class FilesystemObjectStorage(ObjectStorage):
    """Object storage backed by a local directory tree."""

    def __init__(self, root: Path, *, logger: EnhancedLogger | None = None) -> None:
        self.root = Path(root)
        self.logger = logger or get_logger()

    def _path_for_key(self, key: str) -> Path:
        return self.root / Path(key)

    def _relative_key(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    @contextmanager
    def open_text(self, key: str, *, encoding: str = "utf-8-sig") -> Iterator[TextStreamSource]:
        path = self._path_for_key(key)
        validate_existing_file(path, logger=self.logger, error_type=FileError)
        try:
            with path.open(encoding=encoding) as stream:
                yield TextStreamSource(stream, source_name=key)
        except OSError as e:
            self.logger.log_and_raise(FileError(f"Could not open {key}: {e}"))

    def read_bytes(self, key: str) -> bytes | None:
        path = self._path_for_key(key)
        if not path.is_file():
            return None
        try:
            return path.read_bytes()
        except OSError as e:
            self.logger.log_and_raise(FileError(f"Could not read {key}: {e}"))

    def write_bytes(self, key: str, data: bytes, *, overwrite: bool = False) -> None:
        path = self._path_for_key(key)
        ensure_destination_parent(path, logger=self.logger, error_type=FileError)
        if path.exists() and not overwrite:
            self.logger.log_and_raise(FileError(f"Object already exists: {key}"))
        try:
            path.write_bytes(data)
        except OSError as e:
            self.logger.log_and_raise(FileError(f"Could not write {key}: {e}"))

    def write_text_atomic(
        self,
        key: str,
        text: str,
        *,
        encoding: str = "utf-8",
        overwrite: bool = False,
    ) -> None:
        temp_key = ObjectStorage.sibling_temp_key(key)
        try:
            self.write_bytes(temp_key, text.encode(encoding), overwrite=True)
            self.move(temp_key, key, overwrite=overwrite)
        except FileError:
            self.delete(temp_key, missing_ok=True)
            raise

    def list_keys(
        self,
        prefix: str = "",
        *,
        recursive: bool = True,
        suffix: str | None = None,
        glob_pattern: str | None = None,
    ) -> list[str]:
        search_root = self._path_for_key(prefix) if prefix else self.root
        if not search_root.exists():
            return []
        try:
            paths = discover_files(
                search_root,
                recursive=recursive,
                suffix=suffix,
                glob_pattern=glob_pattern,
                logger=self.logger,
            )
        except FileError:
            return []
        return [self._relative_key(path) for path in paths]

    def move(self, source_key: str, dest_key: str, *, overwrite: bool = False) -> None:
        source = self._path_for_key(source_key)
        destination = self._path_for_key(dest_key)
        try:
            move_file(
                source,
                destination,
                overwrite=overwrite,
                logger=self.logger,
                error_type=FileError,
            )
        except FileError as e:
            self.logger.log_and_raise(FileError(f"Could not move {source_key} to {dest_key}: {e}"))

    def delete(self, key: str, *, missing_ok: bool = True) -> None:
        path = self._path_for_key(key)
        remove_file(path, missing_ok=missing_ok, logger=self.logger, error_type=FileError)

    def delete_keys(self, keys: list[str], *, missing_ok: bool = True) -> None:
        for key in keys:
            self.delete(key, missing_ok=missing_ok)
