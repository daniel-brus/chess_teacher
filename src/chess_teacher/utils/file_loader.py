from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import NamedTuple, TextIO

from chess_teacher.utils.exception_utils import FileReadError
from chess_teacher.utils.file_utils import FileType, validate_existing_file
from chess_teacher.utils.logging_utils import EnhancedLogger, get_logger


class TextStreamSource(NamedTuple):
    """An open text stream and optional label for errors and record metadata."""

    stream: TextIO
    source_name: str | None = None


class FileLoader(ABC):
    """Parse text streams into lists of dictionaries."""

    def __init__(self, file_type: FileType, *, logger: EnhancedLogger | None = None):
        self.file_type = file_type
        self.logger = logger or get_logger()

    def load(
        self,
        stream: TextIO,
        *,
        source_name: str | None = None,
    ) -> list[dict]:
        """Parse records from an open text stream."""
        return self._load(stream, source_name=source_name)

    def load_source(self, source: TextStreamSource) -> list[dict]:
        """Parse records from a :class:`TextStreamSource`."""
        return self._load(source.stream, source_name=source.source_name)

    def load_sources(self, sources: list[TextStreamSource]) -> list[dict]:
        """Parse and concatenate records from multiple text streams."""
        records: list[dict] = []
        for source in sources:
            records.extend(self.load_source(source))
        return records

    def load_path(self, file_path: Path) -> list[dict]:
        """Validate, open, and parse a local file (convenience for path-based callers)."""
        validate_existing_file(file_path, logger=self.logger, error_type=FileReadError)
        try:
            with file_path.open(encoding="utf-8-sig") as stream:
                return self.load_source(TextStreamSource(stream, source_name=file_path.as_posix()))
        except OSError as e:
            self.logger.log_and_raise(FileReadError(f"Could not read {file_path.as_posix()}: {e}"))

    @abstractmethod
    def _load(self, stream: TextIO, *, source_name: str | None = None) -> list[dict]:
        """Parse records from an open text stream."""
        pass


class JsonlLoader(FileLoader):
    """Load newline-delimited JSON from a text stream (e.g. .jsonl, .log)."""

    def __init__(self, *, logger: EnhancedLogger | None = None):
        super().__init__(FileType.JSONL, logger=logger)

    def _load(self, stream: TextIO, *, source_name: str | None = None) -> list[dict]:
        records: list[dict] = []
        source = source_name or "input"
        try:
            for line_no, line in enumerate(stream, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    self.logger.log_and_raise(
                        FileReadError(f"Invalid JSON at line {line_no} in {source}: {e}")
                    )
                if not isinstance(obj, dict):
                    self.logger.log_and_raise(
                        FileReadError(
                            f"Expected JSON object at line {line_no} in {source}, "
                            f"got {type(obj).__name__}"
                        )
                    )
                records.append(obj)
        except OSError as e:
            self.logger.log_and_raise(FileReadError(f"Could not read {source}: {e}"))
        return records


class TsvLoader(FileLoader):
    """Load tab-separated values from a text stream (e.g. .tsv)."""

    def __init__(self, *, logger: EnhancedLogger | None = None):
        super().__init__(FileType.TSV, logger=logger)

    def _load(self, stream: TextIO, *, source_name: str | None = None) -> list[dict]:
        records: list[dict] = []
        source = source_name or "input"
        try:
            reader = csv.reader(stream, delimiter="\t")
            try:
                header = next(reader)
            except StopIteration:
                return records

            if not header or all(not column.strip() for column in header):
                self.logger.log_and_raise(FileReadError(f"Missing or empty TSV header in {source}"))

            column_names = [column.strip() for column in header]
            for line_no, row in enumerate(reader, start=2):
                if not row or all(not cell.strip() for cell in row):
                    continue
                if len(row) != len(column_names):
                    self.logger.log_and_raise(
                        FileReadError(
                            f"Invalid TSV at line {line_no} in {source}: "
                            f"expected {len(column_names)} columns, got {len(row)}"
                        )
                    )
                records.append(dict(zip(column_names, (cell.strip() for cell in row), strict=True)))
        except csv.Error as e:
            self.logger.log_and_raise(FileReadError(f"Invalid TSV in {source}: {e}"))
        except OSError as e:
            self.logger.log_and_raise(FileReadError(f"Could not read {source}: {e}"))
        return records


class FileLoaderFactory:
    """Factory class for creating FileLoader instances."""

    @classmethod
    def get_loader(cls, file_type: FileType, *, logger: EnhancedLogger | None = None) -> FileLoader:
        match file_type:
            case FileType.JSONL:
                return JsonlLoader(logger=logger)
            case FileType.TSV:
                return TsvLoader(logger=logger)
            case _:
                raise ValueError(f"Unsupported file type: {file_type}")
