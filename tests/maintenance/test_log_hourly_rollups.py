"""Tests for log analytics rollup transformations and helpers."""

from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl

from chess_teacher.maintenance.dataclasses import ExceptionHourlyCount, LogLevelHourlyCount
from chess_teacher.maintenance.pipeline_steps import (
    DeleteOldS3LogFilesStep,
    LoadRawLogsStep,
)
from chess_teacher.maintenance.transformations import (
    NO_EXC_TYPE_LABEL,
    AggregateExceptionHourlyTransformation,
    AggregateLogLevelHourlyTransformation,
)

_SOURCE_FILE = "logs/python/buffer/closed/2026/06/07/pod-a/app-120000Z.log"
_BUCKET_START = datetime(2026, 6, 7, 12, 0, tzinfo=UTC)


class TestLogStorageHostname:
    def test_relative_closed_log_key(self) -> None:
        assert (
            LoadRawLogsStep.relative_closed_log_key(_SOURCE_FILE)
            == "2026/06/07/pod-a/app-120000Z.log"
        )

    def test_parse_closed_log_hostname(self) -> None:
        assert LoadRawLogsStep.parse_closed_log_hostname(_SOURCE_FILE) == "pod-a"

    def test_parse_closed_log_hostname_unknown(self) -> None:
        assert (
            LoadRawLogsStep.parse_closed_log_hostname("logs/python/quarantine/bad.log") == "unknown"
        )

    def test_parse_closed_log_path_date(self) -> None:
        relative = LoadRawLogsStep.relative_closed_log_key(_SOURCE_FILE)
        assert DeleteOldS3LogFilesStep.parse_closed_log_path_date(relative) == date(2026, 6, 7)


class TestAggregateLogLevelHourlyTransformation:
    def test_aggregates_rows_into_hourly_counts(self) -> None:
        df = pl.DataFrame({
            "ts": [
                datetime(2026, 6, 7, 12, 30, tzinfo=UTC),
                datetime(2026, 6, 7, 12, 45, tzinfo=UTC),
                datetime(2026, 6, 7, 13, 5, tzinfo=UTC),
            ],
            "environment": ["test", "test", "test"],
            "level": ["INFO", "INFO", "ERROR"],
            "logger": ["chess_teacher.foo", "chess_teacher.foo", "chess_teacher.foo"],
            "source_file": [_SOURCE_FILE, _SOURCE_FILE, _SOURCE_FILE],
        })

        result = AggregateLogLevelHourlyTransformation().transform(df)

        assert result.height == 2
        info_row = result.filter(pl.col("level") == "INFO").row(0, named=True)
        assert info_row["log_count"] == 2
        assert info_row["hostname"] == "pod-a"
        assert info_row["bucket_start"] == _BUCKET_START

        error_row = result.filter(pl.col("level") == "ERROR").row(0, named=True)
        assert error_row["log_count"] == 1
        assert error_row["bucket_start"] == datetime(2026, 6, 7, 13, 0, tzinfo=UTC)

    def test_empty_frame_returns_empty_schema(self) -> None:
        result = AggregateLogLevelHourlyTransformation().transform(pl.DataFrame())
        assert result.height == 0
        assert "log_count" in result.columns


class TestAggregateExceptionHourlyTransformation:
    def test_aggregates_exception_rows(self) -> None:
        df = pl.DataFrame({
            "ts": [
                datetime(2026, 6, 7, 12, 10, tzinfo=UTC),
                datetime(2026, 6, 7, 12, 20, tzinfo=UTC),
                datetime(2026, 6, 7, 12, 30, tzinfo=UTC),
            ],
            "environment": ["test", "test", "test"],
            "level": ["ERROR", "ERROR", "WARNING"],
            "exc_type": ["DatabaseError", "DatabaseError", None],
        })

        result = AggregateExceptionHourlyTransformation().transform(df)

        assert result.height == 2
        db_row = result.filter(pl.col("exc_type") == "DatabaseError").row(0, named=True)
        assert db_row["exception_count"] == 2
        assert db_row["level"] == "ERROR"
        assert db_row["bucket_start"] == _BUCKET_START

        warning_row = result.filter(pl.col("level") == "WARNING").row(0, named=True)
        assert warning_row["exception_count"] == 1
        assert warning_row["exc_type"] == NO_EXC_TYPE_LABEL

    def test_aggregates_rows_without_exc_type_column(self) -> None:
        df = pl.DataFrame({
            "ts": [datetime(2026, 6, 7, 12, 10, tzinfo=UTC)],
            "environment": ["test"],
            "level": ["ERROR"],
        })
        result = AggregateExceptionHourlyTransformation().transform(df)
        assert result.height == 1
        assert result["exc_type"][0] == NO_EXC_TYPE_LABEL
        assert result["exception_count"][0] == 1

    def test_empty_frame_returns_empty_schema(self) -> None:
        result = AggregateExceptionHourlyTransformation().transform(pl.DataFrame())
        assert result.height == 0
        assert "exception_count" in result.columns


class TestAnalyticsTableMetadataSync:
    def test_log_level_hourly_count_metadata_sync(self) -> None:
        LogLevelHourlyCount.assert_metadata_sync()

    def test_exception_hourly_count_metadata_sync(self) -> None:
        ExceptionHourlyCount.assert_metadata_sync()
