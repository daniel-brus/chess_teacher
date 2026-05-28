from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

from chess_teacher.utils.exception_utils import FileError, FileWriteError
from chess_teacher.utils.file_utils import (
    FileType,
    check_destination_for_write,
    ensure_destination_parent,
    move_file,
    remove_file,
    validate_existing_file,
)
from chess_teacher.utils.logging_utils import EnhancedLogger, get_logger


class FileWriter(ABC):
    """Write data to a file atomically (temp file, then replace)."""

    def __init__(
        self,
        file_type: FileType,
        *,
        mkdir: bool = True,
        overwrite: bool = False,
        logger: EnhancedLogger | None = None,
    ):
        self.file_type = file_type
        self.mkdir = mkdir
        self.overwrite = overwrite
        self.logger = logger or get_logger()

    def write(self, data: list[dict], file_path: Path) -> None:
        """Write data to a file."""
        self._validate_data(data)
        self._validate_target_path(file_path)
        self._write_atomic(data, file_path)

    def _validate_data(self, data: list[dict]) -> None:
        """Validate records before writing."""
        if not isinstance(data, list):
            self.logger.log_and_raise(
                FileWriteError(f"Expected list of dicts, got {type(data).__name__}")
            )
        for index, record in enumerate(data):
            if not isinstance(record, dict):
                self.logger.log_and_raise(
                    FileWriteError(f"Expected dict at index {index}, got {type(record).__name__}")
                )

    def _validate_target_path(self, file_path: Path) -> None:
        """Validate the target path and ensure the parent directory exists."""
        if file_path.exists():
            validate_existing_file(
                file_path,
                label="Target",
                logger=self.logger,
                error_type=FileWriteError,
            )
        ensure_destination_parent(
            file_path,
            mkdir=self.mkdir,
            logger=self.logger,
            error_type=FileWriteError,
        )
        check_destination_for_write(
            file_path,
            overwrite=self.overwrite,
            logger=self.logger,
            error_type=FileWriteError,
        )

    def _write_atomic(self, data: list[dict], file_path: Path) -> None:
        """Write to a temporary file, then move it into place atomically."""
        tmp_path = file_path.with_name(file_path.name + ".tmp")
        try:
            self._write(data, tmp_path)
            move_file(
                tmp_path,
                file_path,
                overwrite=self.overwrite,
                mkdir=False,
                logger=self.logger,
            )
        except FileWriteError:
            remove_file(tmp_path, missing_ok=True, logger=self.logger)
            raise
        except FileError as e:
            remove_file(tmp_path, missing_ok=True, logger=self.logger)
            self.logger.log_and_raise(
                FileWriteError(f"Could not write atomically to {file_path}: {e}")
            )

    @abstractmethod
    def _write(self, data: list[dict], file_path: Path) -> None:
        """Write data to the given path (typically a temporary file)."""
        pass


class JsonlWriter(FileWriter):
    """Write newline-delimited JSON files (e.g. .jsonl, .log)."""

    def __init__(
        self,
        *,
        mkdir: bool = True,
        overwrite: bool = False,
        logger: EnhancedLogger | None = None,
    ):
        super().__init__(
            FileType.JSONL,
            mkdir=mkdir,
            overwrite=overwrite,
            logger=logger,
        )

    def _write(self, data: list[dict], file_path: Path) -> None:
        try:
            with file_path.open("w", encoding="utf-8") as f:
                for index, record in enumerate(data):
                    try:
                        line = json.dumps(record, ensure_ascii=False)
                    except TypeError as e:
                        self.logger.log_and_raise(
                            FileWriteError(f"Record at index {index} is not JSON-serializable: {e}")
                        )
                    f.write(line + "\n")
        except OSError as e:
            self.logger.log_and_raise(FileWriteError(f"Could not write to {file_path}: {e}"))


class FileWriterFactory:
    """Factory class for creating FileWriter instances."""

    @classmethod
    def get_writer(
        cls,
        file_type: FileType,
        *,
        mkdir: bool = True,
        overwrite: bool = False,
        logger: EnhancedLogger | None = None,
    ) -> FileWriter:
        match file_type:
            case FileType.JSONL:
                return JsonlWriter(
                    mkdir=mkdir,
                    overwrite=overwrite,
                    logger=logger,
                )
            case _:
                raise ValueError(f"Unsupported file type: {file_type}")
