from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

from chess_teacher.utils.exception_utils import FileReadError
from chess_teacher.utils.file_utils import FileType, validate_existing_file
from chess_teacher.utils.logging_utils import EnhancedLogger, get_logger


class FileLoader(ABC):
    """Load data from a file into a list of dictionaries."""

    def __init__(self, file_type: FileType, *, logger: EnhancedLogger | None = None):
        self.file_type = file_type
        self.logger = logger or get_logger()

    def load(self, file_path: Path) -> list[dict]:
        """Load data from the file into a list of dictionaries."""
        validate_existing_file(file_path, logger=self.logger, error_type=FileReadError)
        return self._load(file_path)

    @abstractmethod
    def _load(self, file_path: Path) -> list[dict]:
        """Load data from the file into a list of dictionaries."""
        pass


class JsonlLoader(FileLoader):
    """Load newline-delimited JSON files (e.g. .jsonl, .log)."""

    def __init__(self, *, logger: EnhancedLogger | None = None):
        super().__init__(FileType.JSONL, logger=logger)

    def _load(self, file_path: Path) -> list[dict]:
        records: list[dict] = []
        try:
            with file_path.open(encoding="utf-8-sig") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as e:
                        self.logger.log_and_raise(
                            FileReadError(f"Invalid JSON at line {line_no} in {file_path}: {e}")
                        )
                    if not isinstance(obj, dict):
                        self.logger.log_and_raise(
                            FileReadError(
                                f"Expected JSON object at line {line_no} in {file_path}, "
                                f"got {type(obj).__name__}"
                            )
                        )
                    records.append(obj)
        except OSError as e:
            self.logger.log_and_raise(FileReadError(f"Could not read {file_path}: {e}"))
        return records


class FileLoaderFactory:
    """Factory class for creating FileLoader instances."""

    @classmethod
    def get_loader(cls, file_type: FileType, *, logger: EnhancedLogger | None = None) -> FileLoader:
        match file_type:
            case FileType.JSONL:
                return JsonlLoader(logger=logger)
            case _:
                raise ValueError(f"Unsupported file type: {file_type}")
