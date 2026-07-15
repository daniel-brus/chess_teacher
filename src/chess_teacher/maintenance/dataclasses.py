from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from chess_teacher.utils.table_data_class import TableDataClass

_METADATA_PATH = Path(__file__).parent / "metadata.yml"


@dataclass(frozen=True)
class RawLog(TableDataClass):
    log_id: str
    ts: datetime
    level: str
    logger: str
    msg: str
    environment: str
    source_file: str
    loaded_at: datetime
    exc_type: str | None = None
    exc_msg: str | None = None
    traceback: str | None = None

    @classmethod
    def get_yaml_path(cls) -> Path:
        return _METADATA_PATH

    @classmethod
    def get_key(cls) -> str:
        return "raw_logs"

    @classmethod
    def get_id_hash_columns(cls) -> tuple[str, ...]:
        return ()


@dataclass(frozen=True)
class WarningErrorLog(TableDataClass):
    log_id: str
    ts: datetime
    level: str
    logger: str
    msg: str
    environment: str
    source_file: str
    loaded_at: datetime
    exc_type: str | None = None
    exc_msg: str | None = None
    traceback: str | None = None

    @classmethod
    def get_yaml_path(cls) -> Path:
        return _METADATA_PATH

    @classmethod
    def get_key(cls) -> str:
        return "warning_error_logs"

    @classmethod
    def get_id_hash_columns(cls) -> tuple[str, ...]:
        return ()


@dataclass(frozen=True)
class LogLevelHourlyCount(TableDataClass):
    bucket_start: datetime
    environment: str
    level: str
    logger: str
    hostname: str
    log_count: int

    @classmethod
    def get_yaml_path(cls) -> Path:
        return _METADATA_PATH

    @classmethod
    def get_key(cls) -> str:
        return "log_level_hourly_counts"

    @classmethod
    def get_id_hash_columns(cls) -> tuple[str, ...]:
        return ()


@dataclass(frozen=True)
class ExceptionHourlyCount(TableDataClass):
    bucket_start: datetime
    environment: str
    level: str
    exc_type: str
    exception_count: int

    @classmethod
    def get_yaml_path(cls) -> Path:
        return _METADATA_PATH

    @classmethod
    def get_key(cls) -> str:
        return "exception_hourly_counts"

    @classmethod
    def get_id_hash_columns(cls) -> tuple[str, ...]:
        return ()
