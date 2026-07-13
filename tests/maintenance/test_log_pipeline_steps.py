"""Tests for maintenance log ingestion pipeline steps."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from unittest.mock import MagicMock

import polars as pl
import pytest

from chess_teacher.maintenance.dataclasses import RawLog, WarningErrorLog
from chess_teacher.maintenance.pipeline_steps import (
    DeleteOldS3LogFilesStep,
    LoadRawLogsStep,
    PromoteWarningErrorLogsStep,
)
from chess_teacher.maintenance.transformations import (
    FilterWarningErrorLevelsTransformation,
    ParseLogTimestampColumnsTransformation,
)
from chess_teacher.utils.db.client import WriteResult, WriteStrategy
from chess_teacher.utils.object_storage.filesystem import FilesystemObjectStorage
from chess_teacher.utils.pipeline_utils.pipeline_base import PipelineContext


def _log_line(
    *,
    log_id: str,
    level: str = "INFO",
    msg: str = "hello",
) -> dict[str, str]:
    return {
        "ts": "2026-06-07T12:00:00+00:00",
        "level": level,
        "logger": "test.logger",
        "msg": msg,
        "log_id": log_id,
        "environment": "test",
    }


def _closed_log_key(
    *,
    year: int = 2026,
    month: int = 6,
    day: int = 7,
    hostname: str = "test-host",
    filename: str = "app-120000Z.log",
) -> str:
    return f"{LoadRawLogsStep.CLOSED_LOG_STORAGE_PREFIX}/{year:04d}/{month:02d}/{day:02d}/{hostname}/{filename}"


class TestLogStorageHelpers:
    def test_parse_closed_log_path_date(self) -> None:
        relative = "2026/06/07/test-host/app-120000Z.log"
        assert DeleteOldS3LogFilesStep.parse_closed_log_path_date(relative) == date(2026, 6, 7)

    def test_parse_closed_log_path_date_invalid(self) -> None:
        assert DeleteOldS3LogFilesStep.parse_closed_log_path_date("bad/path.log") is None


class TestLogTransformations:
    def test_parse_log_timestamp_columns(self) -> None:
        df = pl.DataFrame({"ts": ["2026-06-07T12:00:00+00:00"], "level": ["INFO"]})
        result = ParseLogTimestampColumnsTransformation().transform(df)
        assert result.schema["ts"] == pl.Datetime(time_zone="UTC")

    def test_filter_warning_error_levels(self) -> None:
        df = pl.DataFrame({
            "level": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            "log_id": ["a", "b", "c", "d", "e"],
        })
        result = FilterWarningErrorLevelsTransformation().transform(df)
        assert result["level"].to_list() == ["WARNING", "ERROR", "CRITICAL"]


class TestLoadRawLogsStep:
    def test_loads_log_files_into_dataframe(
        self,
        isolate_raw_storage: FilesystemObjectStorage,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        storage = isolate_raw_storage
        key = _closed_log_key()
        payload = json.dumps(_log_line(log_id="log-1")) + "\n"
        storage.write_bytes(key, payload.encode("utf-8"))

        step = LoadRawLogsStep(storage=storage)
        df = step._load_records(MagicMock(), PipelineContext())
        assert df.height == 1
        assert df["log_id"][0] == "log-1"
        assert df["_source_file"][0] == key

    def test_quarantines_unparseable_file(
        self,
        isolate_raw_storage: FilesystemObjectStorage,
    ) -> None:
        storage = isolate_raw_storage
        key = _closed_log_key(filename="bad.log")
        storage.write_bytes(key, b"not json\n")

        step = LoadRawLogsStep(storage=storage)
        df = step._load_records(MagicMock(), PipelineContext())
        assert df.height == 0

        quarantined = storage.list_keys(
            LoadRawLogsStep.QUARANTINE_LOG_STORAGE_PREFIX, recursive=True
        )
        assert len(quarantined) == 1
        assert storage.read_bytes(key) is None


class TestPromoteWarningErrorLogsStep:
    def test_promotes_warning_and_error_rows(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rows = []
        for log_id, level in [("info-1", "INFO"), ("warn-1", "WARNING"), ("err-1", "ERROR")]:
            row = _log_line(log_id=log_id, level=level)
            row["source_file"] = "logs/python/buffer/closed/2026/06/07/host/app.log"
            row["loaded_at"] = datetime(2026, 6, 7, 12, 0, tzinfo=UTC)
            row["ts"] = datetime(2026, 6, 7, 12, 0, tzinfo=UTC)
            rows.append(row)
        source_df = pl.DataFrame(rows)

        db_client = MagicMock()
        db_client.ensure_metadata.return_value = None
        db_client.table_exists.return_value = True
        db_client.engine.execute_parameterized_query.return_value = []

        step = PromoteWarningErrorLogsStep()
        step._incremental_filter.db_client = db_client

        saved_frames: list[pl.DataFrame] = []

        def capture_save(
            _db_client: MagicMock,
            _table_metadata: object,
            data: pl.DataFrame,
        ) -> WriteResult:
            saved_frames.append(data)
            return WriteResult(strategy=WriteStrategy.INSERT_IGNORE, rows_inserted=data.height)

        monkeypatch.setattr(step, "_load_records", lambda _db, _ctx: source_df)
        monkeypatch.setattr(step, "_save_records", capture_save)

        step.run(db_client, PipelineContext())

        assert len(saved_frames) == 1
        assert saved_frames[0].height == 2
        assert set(saved_frames[0]["log_id"].to_list()) == {"warn-1", "err-1"}


class TestDeleteOldS3LogFilesStep:
    def test_deletes_old_log_files_by_path_date(
        self,
        isolate_raw_storage: FilesystemObjectStorage,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        storage = isolate_raw_storage
        old_key = _closed_log_key(year=2020, month=1, day=1)
        recent_key = _closed_log_key(year=2026, month=6, day=7)
        storage.write_bytes(old_key, b"{}\n")
        storage.write_bytes(recent_key, b"{}\n")

        fixed_now = datetime(2026, 6, 7, 12, 0, tzinfo=UTC)
        monkeypatch.setattr(
            "chess_teacher.maintenance.pipeline_steps.get_current_datetime",
            lambda: fixed_now,
        )

        step = DeleteOldS3LogFilesStep(storage=storage)
        step.run(MagicMock(), PipelineContext())

        assert storage.read_bytes(old_key) is None
        assert storage.read_bytes(recent_key) is not None


class TestTableMetadataSync:
    def test_raw_log_metadata_sync(self) -> None:
        RawLog.assert_metadata_sync()

    def test_warning_error_log_metadata_sync(self) -> None:
        WarningErrorLog.assert_metadata_sync()
