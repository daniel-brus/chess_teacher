"""Tests for admin log analytics helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from chess_teacher.maintenance.log_analytics import (
    ERROR_PLUS_LEVELS,
    SEVERITY_LEVELS,
    LogDashboardFilters,
    PeriodDelta,
    apply_exception_filters,
    apply_log_level_filters,
    build_log_volume_timeseries,
    build_warning_error_log_where,
    compute_health_metrics,
    compute_log_dashboard_summary,
    compute_period_delta,
    describe_filter_scope,
    effective_severity_levels,
    format_hours_ago,
    format_log_exception_summary,
    format_period_delta_detail,
    get_bucket_bounds,
    get_data_freshness,
    recent_warning_error_log_rows,
    summarize_by_subsystem,
    top_error_loggers,
    top_exception_types,
    top_hosts_by_errors,
    top_loggers,
    truncate_log_message,
)
from chess_teacher.maintenance.transformations import NO_EXC_TYPE_LABEL

_BUCKET = datetime(2026, 6, 7, 12, 0, tzinfo=UTC)
_LEVEL_ROWS = pl.DataFrame({
    "bucket_start": [_BUCKET, _BUCKET, _BUCKET],
    "environment": ["prod", "prod", "prod"],
    "level": ["INFO", "ERROR", "ERROR"],
    "logger": ["app.main", "app.db", "app.db"],
    "hostname": ["pod-a", "pod-a", "pod-b"],
    "log_count": [100, 5, 3],
})
_EXCEPTION_ROWS = pl.DataFrame({
    "bucket_start": [_BUCKET, _BUCKET],
    "environment": ["prod", "prod"],
    "level": ["ERROR", "WARNING"],
    "exc_type": ["DatabaseError", NO_EXC_TYPE_LABEL],
    "exception_count": [5, 2],
})

_LATEST = datetime(2026, 6, 7, 12, 0, tzinfo=UTC)
_PREV_DAY = datetime(2026, 6, 6, 12, 0, tzinfo=UTC)
_MULTI_BUCKET_LEVELS = pl.DataFrame({
    "bucket_start": [_PREV_DAY, _PREV_DAY, _LATEST, _LATEST, _LATEST, _LATEST],
    "environment": ["prod"] * 6,
    "level": ["ERROR", "INFO", "INFO", "ERROR", "CRITICAL", "INFO"],
    "logger": [
        "chess_teacher.maintenance.foo",
        "app.main",
        "app.main",
        "chess_teacher.maintenance.bar",
        "chess_teacher.maintenance.bar",
        "streamlit_pages.admin",
    ],
    "hostname": ["pod-a", "pod-a", "pod-a", "pod-a", "pod-a", "pod-b"],
    "log_count": [10, 500, 500, 20, 2, 500],
})
_MULTI_BUCKET_EXCEPTIONS = pl.DataFrame({
    "bucket_start": [_LATEST],
    "environment": ["prod"],
    "level": ["ERROR"],
    "exc_type": ["DatabaseError"],
    "exception_count": [3],
})


class TestLogDashboardFilters:
    def test_get_bucket_bounds(self) -> None:
        assert get_bucket_bounds(_LEVEL_ROWS) == (_BUCKET.date(), _BUCKET.date())

    def test_apply_log_level_filters_by_level(self) -> None:
        filtered = apply_log_level_filters(
            _LEVEL_ROWS,
            LogDashboardFilters(levels=frozenset({"ERROR"})),
        )
        assert filtered.height == 2
        assert filtered["log_count"].sum() == 8

    def test_apply_exception_filters_by_environment(self) -> None:
        filtered = apply_exception_filters(
            _EXCEPTION_ROWS,
            LogDashboardFilters(environments=frozenset({"staging"})),
        )
        assert filtered.is_empty()


class TestApplyExceptionFilters:
    def test_apply_exception_filters_by_level(self) -> None:
        filtered = apply_exception_filters(
            _EXCEPTION_ROWS,
            LogDashboardFilters(levels=frozenset({"ERROR"})),
        )
        assert filtered.height == 1
        assert filtered["level"][0] == "ERROR"


class TestLogDashboardSummary:
    def test_compute_summary(self) -> None:
        summary = compute_log_dashboard_summary(_LEVEL_ROWS, _EXCEPTION_ROWS)
        assert summary.total_lines == 108
        assert summary.info_lines == 100
        assert summary.error_lines == 8
        assert summary.distinct_loggers == 2
        assert summary.distinct_hosts == 2
        assert summary.warning_error_events == 7
        assert summary.traced_exceptions == 5

    def test_compute_summary_empty(self) -> None:
        summary = compute_log_dashboard_summary(
            pl.DataFrame(schema=_LEVEL_ROWS.schema),
            pl.DataFrame(schema=_EXCEPTION_ROWS.schema),
        )
        assert summary.total_lines == 0
        assert summary.traced_exceptions == 0

    def test_build_log_volume_timeseries(self) -> None:
        series = build_log_volume_timeseries(_LEVEL_ROWS)
        assert series.height == 2
        assert series.filter(pl.col("level") == "INFO")["log_count"][0] == 100

    def test_top_loggers_and_exceptions(self) -> None:
        assert top_loggers(_LEVEL_ROWS) == [("app.main", 100), ("app.db", 8)]
        assert top_exception_types(_EXCEPTION_ROWS) == [("DatabaseError", 5)]


class TestPeriodDelta:
    def test_compute_period_delta_error_plus(self) -> None:
        delta = compute_period_delta(
            _MULTI_BUCKET_LEVELS,
            count_col="log_count",
            latest_bucket=_LATEST,
            level_col="level",
            levels=ERROR_PLUS_LEVELS,
        )
        assert delta.current == 22
        assert delta.previous == 10
        assert delta.delta == 12
        assert delta.delta_pct == pytest.approx(120.0)


class TestDataFreshness:
    def test_get_data_freshness_not_stale(self) -> None:
        freshness = get_data_freshness(
            _MULTI_BUCKET_LEVELS,
            _MULTI_BUCKET_EXCEPTIONS,
            reference_now=_LATEST + timedelta(hours=2),
        )
        assert freshness.latest_bucket == _LATEST
        assert freshness.is_stale is False
        assert freshness.hours_since_latest == pytest.approx(2.0)

    def test_get_data_freshness_stale(self) -> None:
        freshness = get_data_freshness(
            _MULTI_BUCKET_LEVELS,
            _MULTI_BUCKET_EXCEPTIONS,
            reference_now=_LATEST + timedelta(hours=30),
        )
        assert freshness.is_stale is True


class TestHealthMetrics:
    def test_compute_health_metrics(self) -> None:
        health = compute_health_metrics(
            _MULTI_BUCKET_LEVELS,
            _MULTI_BUCKET_EXCEPTIONS,
            reference_now=_LATEST,
        )
        assert health.error_plus_24h.current == 22
        assert health.error_plus_24h.previous == 10
        assert health.error_plus_24h.delta == 12
        assert health.critical_24h == 2
        assert health.error_rate_pct == pytest.approx(2.2)
        assert health.freshness.latest_bucket == _LATEST

    def test_compute_health_metrics_scoped_to_environment(self) -> None:
        multi_env = _MULTI_BUCKET_LEVELS.with_columns(
            pl
            .when(pl.col("bucket_start") == _LATEST)
            .then(pl.lit("staging"))
            .otherwise(pl.col("environment"))
            .alias("environment"),
        )
        health = compute_health_metrics(
            multi_env,
            _MULTI_BUCKET_EXCEPTIONS,
            reference_now=_LATEST,
            environments=frozenset({"prod"}),
        )
        assert health.error_plus_24h.current == 10
        assert health.critical_24h == 0


class TestTopErrorAggregations:
    def test_top_error_loggers(self) -> None:
        rows = top_error_loggers(_MULTI_BUCKET_LEVELS)
        assert all(logger != "app.main" for logger, _ in rows)
        assert rows[0] == ("chess_teacher.maintenance.bar", 22)

    def test_top_hosts_by_errors(self) -> None:
        rows = top_hosts_by_errors(_MULTI_BUCKET_LEVELS)
        assert rows == [("pod-a", 32)]


class TestSeverityTimeseries:
    def test_build_log_volume_timeseries_levels_filter(self) -> None:
        series = build_log_volume_timeseries(_LEVEL_ROWS, levels=SEVERITY_LEVELS)
        assert series.height == 1
        assert "INFO" not in series["level"].to_list()


class TestSubsystemSummary:
    def test_summarize_by_subsystem(self) -> None:
        rows = summarize_by_subsystem(_MULTI_BUCKET_LEVELS)
        by_name = {row.subsystem: row for row in rows}
        assert by_name["Maintenance"].error_plus_lines == 32
        assert by_name["Streamlit"].total_lines == 500
        assert "Other" in by_name


class TestFilterScopeDescription:
    def test_describe_filter_scope(self) -> None:
        text = describe_filter_scope(
            LogDashboardFilters(
                date_from=_PREV_DAY.date(),
                date_to=_LATEST.date(),
                environments=frozenset({"prod"}),
                levels=frozenset({"ERROR", "CRITICAL"}),
            ),
            bounds=(_PREV_DAY.date(), _LATEST.date()),
            all_environments=["prod", "staging"],
            all_levels=["INFO", "ERROR", "CRITICAL"],
        )
        assert "prod" in text
        assert "Jun" in text
        assert "ERROR, CRITICAL" in text

    def test_describe_filter_scope_single_environment(self) -> None:
        text = describe_filter_scope(
            LogDashboardFilters(environments=frozenset({"prod"})),
            bounds=(_PREV_DAY.date(), _LATEST.date()),
            all_environments=["prod"],
            all_levels=["INFO", "ERROR"],
        )
        assert text.startswith("prod ·")
        assert "all environments" not in text


class TestRecentWarningErrorLogs:
    def test_effective_severity_levels_respects_filter(self) -> None:
        assert effective_severity_levels(LogDashboardFilters()) == SEVERITY_LEVELS
        assert effective_severity_levels(
            LogDashboardFilters(levels=frozenset({"ERROR"}))
        ) == frozenset({"ERROR"})
        assert (
            effective_severity_levels(LogDashboardFilters(levels=frozenset({"INFO"})))
            == frozenset()
        )

    def test_build_warning_error_log_where(self) -> None:
        where = build_warning_error_log_where(
            LogDashboardFilters(
                date_from=_PREV_DAY.date(),
                date_to=_LATEST.date(),
                environments=frozenset({"prod"}),
                levels=frozenset({"ERROR", "WARNING"}),
            )
        )
        assert where is not None
        assert "level" in where
        assert "'ERROR'" in where
        assert "'prod'" in where

    def test_truncate_log_message(self) -> None:
        assert truncate_log_message("short") == "short"
        assert truncate_log_message("x" * 300).endswith("…")
        assert len(truncate_log_message("x" * 300)) == 240

    def test_format_log_exception_summary(self) -> None:
        assert format_log_exception_summary("ValueError", "bad input") == "ValueError: bad input"
        assert format_log_exception_summary(None, "ignored") is None

    def test_recent_warning_error_log_rows(self) -> None:
        logs = pl.DataFrame({
            "ts": [_LATEST],
            "level": ["ERROR"],
            "logger": ["app.main"],
            "msg": ["Something failed"],
            "environment": ["prod"],
            "exc_type": ["RuntimeError"],
            "exc_msg": ["boom"],
        })
        rows = recent_warning_error_log_rows(logs)
        assert rows[0]["Level"] == "ERROR"
        assert rows[0]["Message"] == "Something failed"
        assert rows[0]["Exception"] == "RuntimeError: boom"


class TestFormatHelpers:
    def test_format_hours_ago(self) -> None:
        assert format_hours_ago(None) == "—"
        assert format_hours_ago(0.5) == "<1h ago"
        assert format_hours_ago(3.2) == "3h ago"

    def test_format_period_delta_detail_positive(self) -> None:
        detail = format_period_delta_detail(
            PeriodDelta(current=15, previous=10, delta=5, delta_pct=50.0),
        )
        assert detail == "+5 (+50%) vs prior 24h"

    def test_format_period_delta_detail_negative(self) -> None:
        detail = format_period_delta_detail(
            PeriodDelta(current=8, previous=20, delta=-12, delta_pct=-60.0),
        )
        assert detail == "-12 (-60%) vs prior 24h"

    def test_format_period_delta_detail_zero(self) -> None:
        detail = format_period_delta_detail(
            PeriodDelta(current=0, previous=0, delta=0, delta_pct=None),
        )
        assert detail == "No prior period data"
