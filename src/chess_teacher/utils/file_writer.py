from __future__ import annotations

import json
from abc import ABC, abstractmethod

from chess_teacher.utils.exception_utils import FileWriteError
from chess_teacher.utils.file_utils import FileType
from chess_teacher.utils.logging_utils import EnhancedLogger, get_logger
from chess_teacher.utils.object_storage.base import ObjectStorage


class FileWriter(ABC):
    """Write data to object storage as newline-delimited records."""

    def __init__(
        self,
        file_type: FileType,
        *,
        overwrite: bool = False,
        logger: EnhancedLogger | None = None,
    ):
        self.file_type = file_type
        self.overwrite = overwrite
        self.logger = logger or get_logger()

    def write(self, data: list[dict], key: str, storage: ObjectStorage) -> None:
        """Write data to a storage key."""
        self._validate_data(data)
        try:
            text = self._serialize(data)
        except FileWriteError:
            raise
        except Exception as e:
            self.logger.log_and_raise(FileWriteError(f"Could not serialize data for {key}: {e}"))
        try:
            storage.write_text_atomic(key, text, overwrite=self.overwrite)
        except Exception as e:
            self.logger.log_and_raise(FileWriteError(f"Could not write to {key}: {e}"))

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

    @abstractmethod
    def _serialize(self, data: list[dict]) -> str:
        """Serialize records to a text payload."""
        pass


class JsonlWriter(FileWriter):
    """Write newline-delimited JSON (e.g. .jsonl, .log)."""

    def __init__(
        self,
        *,
        overwrite: bool = False,
        logger: EnhancedLogger | None = None,
    ):
        super().__init__(
            FileType.JSONL,
            overwrite=overwrite,
            logger=logger,
        )

    def _serialize(self, data: list[dict]) -> str:
        lines: list[str] = []
        for index, record in enumerate(data):
            try:
                lines.append(json.dumps(record, ensure_ascii=False))
            except TypeError as e:
                self.logger.log_and_raise(
                    FileWriteError(f"Record at index {index} is not JSON-serializable: {e}")
                )
        return "\n".join(lines) + ("\n" if lines else "")


class FileWriterFactory:
    """Factory class for creating FileWriter instances."""

    @classmethod
    def get_writer(
        cls,
        file_type: FileType,
        *,
        overwrite: bool = False,
        logger: EnhancedLogger | None = None,
    ) -> FileWriter:
        match file_type:
            case FileType.JSONL:
                return JsonlWriter(
                    overwrite=overwrite,
                    logger=logger,
                )
            case _:
                raise ValueError(f"Unsupported file type: {file_type}")
