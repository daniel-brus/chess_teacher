"""Load and summarize maintenance log aggregate tables for the admin dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

import polars as pl

from chess_teacher.maintenance.dataclasses import (
    ExceptionHourlyCount,
    LogLevelHourlyCount,
    WarningErrorLog,
)
from chess_teacher.maintenance.transformations import NO_EXC_TYPE_LABEL
from chess_teacher.utils.cache_utils import get_cache_client
from chess_teacher.utils.db.client import DatabaseClient
from chess_teacher.utils.general_utils import quote_ident, quote_literal
from chess_teacher.utils.logging import get_logger

DEFAULT_RECENT_LOG_LIMIT = 50
RECENT_LOG_MESSAGE_MAX_LEN = 240

_WARNING_ERROR_LOG_COLUMNS = (
    "ts",
    "level",
    "logger",
    "msg",
    "environment",
    "exc_type",
    "exc_msg",
)

logger = get_logger()

LOG_LEVEL_ORDER = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
LOG_LEVEL_COLORS = {
    "DEBUG": "#9ca3af",
    "INFO": "#3b82f6",
    "WARNING": "#eab308",
    "ERROR": "#ef4444",
    "CRITICAL": "#7f1d1d",
}
SEVERITY_LEVELS = frozenset({"WARNING", "ERROR", "CRITICAL"})
ERROR_PLUS_LEVELS = frozenset({"ERROR", "CRITICAL"})
DEFAULT_FILTER_WINDOW_DAYS = 7
STALE_BUCKET_THRESHOLD_HOURS = 25
LOGGER_SUBSYSTEMS = (
    ("Maintenance", ("chess_teacher.maintenance.",)),
    ("Pipelines", ("chess_teacher.pipelines.",)),
    ("Analytics", ("chess_teacher.analytics.",)),
    ("Bots", ("chess_teacher.bots.",)),
    ("Streamlit", ("streamlit_pages.", "streamlit_utils.")),
    ("Core utils", ("chess_teacher.utils.",)),
)
_LOG_LEVEL_COLUMNS = [
    "bucket_start",
    "environment",
    "level",
    "logger",
    "hostname",
    "log_count",
]
_EXCEPTION_COLUMNS = [
    "bucket_start",
    "environment",
    "level",
    "exc_type",
    "exception_count",
]


def default_filter_date_from(max_date: date) -> date:
    """Default admin dashboard filter start: last N calendar days inclusive."""
    return max_date - timedelta(days=DEFAULT_FILTER_WINDOW_DAYS - 1)


def load_log_level_hourly_counts(db_client: DatabaseClient) -> pl.DataFrame:
    cache = get_cache_client()
    if cache is not None:
        cached = cache.get_log_level_hourly_counts()
        if cached is not None:
            return cached

    db_client.ensure_metadata(LogLevelHourlyCount.get_metadata())
    logger.info("Loading log level hourly counts from database")
    counts = db_client.read(
        LogLevelHourlyCount.get_metadata(),
        columns=_LOG_LEVEL_COLUMNS,
        order_by="bucket_start DESC",
        as_polars=True,
    )
    logger.info("Loaded log level hourly counts rows=%s", counts.height)

    if cache is not None:
        cache.set_log_level_hourly_counts(counts)
    return counts


def load_exception_hourly_counts(db_client: DatabaseClient) -> pl.DataFrame:
    cache = get_cache_client()
    if cache is not None:
        cached = cache.get_exception_hourly_counts()
        if cached is not None:
            return cached

    db_client.ensure_metadata(ExceptionHourlyCount.get_metadata())
    logger.info("Loading exception hourly counts from database")
    counts = db_client.read(
        ExceptionHourlyCount.get_metadata(),
        columns=_EXCEPTION_COLUMNS,
        order_by="bucket_start DESC",
        as_polars=True,
    )
    logger.info("Loaded exception hourly counts rows=%s", counts.height)

    if cache is not None:
        cache.set_exception_hourly_counts(counts)
    return counts


@dataclass(frozen=True)
class LogDashboardFilters:
    date_from: date | None = None
    date_to: date | None = None
    environments: frozenset[str] | None = None
    levels: frozenset[str] | None = None


@dataclass(frozen=True)
class PeriodDelta:
    current: int
    previous: int
    delta: int
    delta_pct: float | None


@dataclass(frozen=True)
class DataFreshness:
    latest_bucket: datetime | None
    hours_since_latest: float | None
    is_stale: bool


@dataclass(frozen=True)
class HealthMetrics:
    error_plus_24h: PeriodDelta
    critical_24h: int
    error_rate_pct: float
    freshness: DataFreshness


@dataclass(frozen=True)
class SubsystemSummary:
    subsystem: str
    total_lines: int
    error_plus_lines: int


def get_bucket_bounds(counts: pl.DataFrame) -> tuple[date, date] | None:
    if counts.is_empty():
        return None
    earliest = counts.select(pl.col("bucket_start").min()).item()
    latest = counts.select(pl.col("bucket_start").max()).item()
    if earliest is None or latest is None:
        return None
    if isinstance(earliest, datetime):
        earliest = earliest.date()
    if isinstance(latest, datetime):
        latest = latest.date()
    return earliest, latest


def apply_log_level_filters(counts: pl.DataFrame, filters: LogDashboardFilters) -> pl.DataFrame:
    filtered = counts
    if filters.environments is not None:
        filtered = filtered.filter(pl.col("environment").is_in(list(filters.environments)))
    if filters.levels is not None:
        filtered = filtered.filter(pl.col("level").is_in(list(filters.levels)))
    if filters.date_from is not None:
        filtered = filtered.filter(pl.col("bucket_start").dt.date() >= filters.date_from)
    if filters.date_to is not None:
        filtered = filtered.filter(pl.col("bucket_start").dt.date() <= filters.date_to)
    return filtered


def apply_exception_filters(counts: pl.DataFrame, filters: LogDashboardFilters) -> pl.DataFrame:
    filtered = counts
    if filters.environments is not None:
        filtered = filtered.filter(pl.col("environment").is_in(list(filters.environments)))
    if filters.levels is not None:
        filtered = filtered.filter(pl.col("level").is_in(list(filters.levels)))
    if filters.date_from is not None:
        filtered = filtered.filter(pl.col("bucket_start").dt.date() >= filters.date_from)
    if filters.date_to is not None:
        filtered = filtered.filter(pl.col("bucket_start").dt.date() <= filters.date_to)
    return filtered


def effective_severity_levels(filters: LogDashboardFilters) -> frozenset[str]:
    """WARNING/ERROR/CRITICAL levels included in the current filter selection."""
    if filters.levels is None:
        return SEVERITY_LEVELS
    return frozenset(level for level in filters.levels if level in SEVERITY_LEVELS)


def _sql_in_clause(ident: str, values: frozenset[str]) -> str:
    literals = ", ".join(quote_literal(value) for value in sorted(values))
    return f"{quote_ident(ident)} IN ({literals})"


def build_warning_error_log_where(filters: LogDashboardFilters) -> str | None:
    """Build a SQL WHERE clause for warning_error_logs matching dashboard filters."""
    levels = effective_severity_levels(filters)
    if not levels:
        return None

    clauses = [_sql_in_clause("level", levels)]
    if filters.environments is not None:
        clauses.append(_sql_in_clause("environment", filters.environments))
    if filters.date_from is not None:
        start = datetime.combine(filters.date_from, time.min, tzinfo=UTC)
        clauses.append(f"{quote_ident('ts')} >= {quote_literal(start.isoformat())}")
    if filters.date_to is not None:
        end_exclusive = datetime.combine(
            filters.date_to + timedelta(days=1),
            time.min,
            tzinfo=UTC,
        )
        clauses.append(f"{quote_ident('ts')} < {quote_literal(end_exclusive.isoformat())}")
    return " AND ".join(clauses)


def load_recent_warning_error_logs(
    db_client: DatabaseClient,
    filters: LogDashboardFilters,
    *,
    limit: int = DEFAULT_RECENT_LOG_LIMIT,
) -> pl.DataFrame:
    """Load the most recent WARNING/ERROR/CRITICAL log lines for the admin dashboard."""
    where = build_warning_error_log_where(filters)
    if where is None:
        return pl.DataFrame(
            schema={
                "ts": pl.Datetime(time_zone="UTC"),
                "level": pl.Utf8,
                "logger": pl.Utf8,
                "msg": pl.Utf8,
                "environment": pl.Utf8,
                "exc_type": pl.Utf8,
                "exc_msg": pl.Utf8,
            }
        )

    db_client.ensure_metadata(WarningErrorLog.get_metadata())
    logger.info("Loading recent warning/error logs for admin dashboard limit=%s", limit)
    rows = db_client.read(
        WarningErrorLog.get_metadata(),
        columns=list(_WARNING_ERROR_LOG_COLUMNS),
        where=where,
        order_by="ts DESC",
        limit=limit,
        as_polars=True,
    )
    logger.info("Loaded recent warning/error logs rows=%s", rows.height)
    return rows


def truncate_log_message(msg: str, *, max_len: int = RECENT_LOG_MESSAGE_MAX_LEN) -> str:
    cleaned = " ".join(msg.split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1] + "…"


def format_log_exception_summary(
    exc_type: str | None,
    exc_msg: str | None,
) -> str | None:
    if not exc_type:
        return None
    if exc_msg:
        return f"{exc_type}: {truncate_log_message(exc_msg, max_len=120)}"
    return exc_type


def recent_warning_error_log_rows(logs: pl.DataFrame) -> list[dict[str, str]]:
    if logs.is_empty():
        return []

    display_rows: list[dict[str, str]] = []
    for row in logs.iter_rows(named=True):
        ts = row["ts"]
        if isinstance(ts, datetime):
            ts_text = ts.strftime("%Y-%m-%d %H:%M:%S UTC")
        else:
            ts_text = str(ts)
        exception = format_log_exception_summary(row.get("exc_type"), row.get("exc_msg"))
        display_rows.append({
            "Time (UTC)": ts_text,
            "Level": str(row["level"]),
            "Logger": str(row["logger"]),
            "Message": truncate_log_message(str(row["msg"])),
            "Exception": exception or "",
        })
    return display_rows


def filter_severity_levels(
    level_counts: pl.DataFrame,
    *,
    levels: frozenset[str] = SEVERITY_LEVELS,
) -> pl.DataFrame:
    if level_counts.is_empty():
        return level_counts
    return level_counts.filter(pl.col("level").is_in(list(levels)))


def _sum_count_in_window(
    counts: pl.DataFrame,
    *,
    count_col: str,
    window_start: datetime,
    window_end: datetime,
    level_col: str | None = None,
    levels: frozenset[str] | None = None,
) -> int:
    if counts.is_empty():
        return 0
    filtered = counts.filter(
        (pl.col("bucket_start") > window_start) & (pl.col("bucket_start") <= window_end)
    )
    if level_col is not None and levels is not None:
        filtered = filtered.filter(pl.col(level_col).is_in(list(levels)))
    if filtered.is_empty():
        return 0
    return int(filtered.select(pl.col(count_col).sum()).item() or 0)


def compute_period_delta(
    counts: pl.DataFrame,
    *,
    count_col: str,
    latest_bucket: datetime,
    period_hours: int = 24,
    level_col: str | None = None,
    levels: frozenset[str] | None = None,
) -> PeriodDelta:
    period = timedelta(hours=period_hours)
    current_start = latest_bucket - period
    previous_start = latest_bucket - (2 * period)
    previous_end = latest_bucket - period

    current = _sum_count_in_window(
        counts,
        count_col=count_col,
        window_start=current_start,
        window_end=latest_bucket,
        level_col=level_col,
        levels=levels,
    )
    previous = _sum_count_in_window(
        counts,
        count_col=count_col,
        window_start=previous_start,
        window_end=previous_end,
        level_col=level_col,
        levels=levels,
    )
    delta = current - previous
    delta_pct = None if previous == 0 else (delta / previous) * 100
    return PeriodDelta(current=current, previous=previous, delta=delta, delta_pct=delta_pct)


def get_data_freshness(
    level_counts: pl.DataFrame,
    exception_counts: pl.DataFrame,
    *,
    reference_now: datetime,
    stale_threshold_hours: int = STALE_BUCKET_THRESHOLD_HOURS,
) -> DataFreshness:
    latest_candidates: list[datetime] = []
    if not level_counts.is_empty():
        level_latest = level_counts.select(pl.col("bucket_start").max()).item()
        if level_latest is not None:
            latest_candidates.append(level_latest)
    if not exception_counts.is_empty():
        exception_latest = exception_counts.select(pl.col("bucket_start").max()).item()
        if exception_latest is not None:
            latest_candidates.append(exception_latest)

    if not latest_candidates:
        return DataFreshness(latest_bucket=None, hours_since_latest=None, is_stale=False)

    latest_bucket = max(latest_candidates)
    hours_since_latest = (reference_now - latest_bucket).total_seconds() / 3600
    return DataFreshness(
        latest_bucket=latest_bucket,
        hours_since_latest=hours_since_latest,
        is_stale=hours_since_latest > stale_threshold_hours,
    )


def _apply_environment_filter(
    counts: pl.DataFrame,
    environments: frozenset[str] | None,
) -> pl.DataFrame:
    if environments is None or counts.is_empty():
        return counts
    return counts.filter(pl.col("environment").is_in(list(environments)))


def compute_health_metrics(
    level_counts: pl.DataFrame,
    exception_counts: pl.DataFrame,
    *,
    reference_now: datetime,
    environments: frozenset[str] | None = None,
) -> HealthMetrics:
    env_levels = _apply_environment_filter(level_counts, environments)
    env_exceptions = _apply_environment_filter(exception_counts, environments)
    freshness = get_data_freshness(env_levels, env_exceptions, reference_now=reference_now)

    if env_levels.is_empty():
        empty_delta = PeriodDelta(current=0, previous=0, delta=0, delta_pct=None)
        return HealthMetrics(
            error_plus_24h=empty_delta,
            critical_24h=0,
            error_rate_pct=0.0,
            freshness=freshness,
        )

    latest_bucket = env_levels.select(pl.col("bucket_start").max()).item()
    assert latest_bucket is not None

    error_plus_24h = compute_period_delta(
        env_levels,
        count_col="log_count",
        latest_bucket=latest_bucket,
        level_col="level",
        levels=ERROR_PLUS_LEVELS,
    )
    critical_24h = _sum_count_in_window(
        env_levels,
        count_col="log_count",
        window_start=latest_bucket - timedelta(hours=24),
        window_end=latest_bucket,
        level_col="level",
        levels=frozenset({"CRITICAL"}),
    )
    total_lines_24h = _sum_count_in_window(
        env_levels,
        count_col="log_count",
        window_start=latest_bucket - timedelta(hours=24),
        window_end=latest_bucket,
    )
    error_rate_pct = (
        0.0
        if total_lines_24h == 0
        else round(
            100 * error_plus_24h.current / total_lines_24h,
            1,
        )
    )

    return HealthMetrics(
        error_plus_24h=error_plus_24h,
        critical_24h=critical_24h,
        error_rate_pct=error_rate_pct,
        freshness=freshness,
    )


@dataclass(frozen=True)
class LogDashboardSummary:
    total_lines: int
    info_lines: int
    warning_lines: int
    error_lines: int
    critical_lines: int
    distinct_loggers: int
    distinct_hosts: int
    warning_error_events: int
    traced_exceptions: int


def compute_log_dashboard_summary(
    level_counts: pl.DataFrame,
    exception_counts: pl.DataFrame,
) -> LogDashboardSummary:
    if level_counts.is_empty():
        return LogDashboardSummary(
            total_lines=0,
            info_lines=0,
            warning_lines=0,
            error_lines=0,
            critical_lines=0,
            distinct_loggers=0,
            distinct_hosts=0,
            warning_error_events=0,
            traced_exceptions=0,
        )

    level_totals = level_counts.group_by("level").agg(pl.col("log_count").sum()).sort("level")
    totals_by_level = {
        row["level"]: int(row["log_count"]) for row in level_totals.iter_rows(named=True)
    }
    traced_exceptions = 0
    warning_error_events = 0
    if not exception_counts.is_empty():
        warning_error_events = int(exception_counts.select(pl.col("exception_count").sum()).item())
        traced_exceptions = int(
            exception_counts
            .filter(pl.col("exc_type") != NO_EXC_TYPE_LABEL)
            .select(pl.col("exception_count").sum())
            .item()
            or 0
        )

    return LogDashboardSummary(
        total_lines=int(level_counts.select(pl.col("log_count").sum()).item()),
        info_lines=totals_by_level.get("INFO", 0),
        warning_lines=totals_by_level.get("WARNING", 0),
        error_lines=totals_by_level.get("ERROR", 0),
        critical_lines=totals_by_level.get("CRITICAL", 0),
        distinct_loggers=level_counts.select(pl.col("logger").n_unique()).item(),
        distinct_hosts=level_counts.select(pl.col("hostname").n_unique()).item(),
        warning_error_events=warning_error_events,
        traced_exceptions=traced_exceptions,
    )


def build_log_volume_timeseries(
    level_counts: pl.DataFrame,
    *,
    levels: frozenset[str] | None = None,
) -> pl.DataFrame:
    if level_counts.is_empty():
        return pl.DataFrame(
            schema={
                "bucket_start": pl.Datetime(time_zone="UTC"),
                "level": pl.Utf8,
                "log_count": pl.Int64,
            }
        )
    filtered = level_counts
    if levels is not None:
        filtered = filtered.filter(pl.col("level").is_in(list(levels)))
    if filtered.is_empty():
        return pl.DataFrame(
            schema={
                "bucket_start": pl.Datetime(time_zone="UTC"),
                "level": pl.Utf8,
                "log_count": pl.Int64,
            }
        )
    return (
        filtered
        .group_by("bucket_start", "level")
        .agg(pl.col("log_count").sum())
        .sort("bucket_start", "level")
    )


def build_exception_volume_timeseries(exception_counts: pl.DataFrame) -> pl.DataFrame:
    if exception_counts.is_empty():
        return pl.DataFrame(
            schema={"bucket_start": pl.Datetime(time_zone="UTC"), "exception_count": pl.Int64}
        )
    return (
        exception_counts
        .group_by("bucket_start")
        .agg(pl.col("exception_count").sum().alias("exception_count"))
        .sort("bucket_start")
    )


def top_loggers(level_counts: pl.DataFrame, *, limit: int = 15) -> list[tuple[str, int]]:
    if level_counts.is_empty():
        return []
    rows = (
        level_counts
        .group_by("logger")
        .agg(pl.col("log_count").sum().alias("log_count"))
        .sort("log_count", descending=True)
        .head(limit)
        .iter_rows(named=True)
    )
    return [(row["logger"], int(row["log_count"])) for row in rows]


def top_error_loggers(level_counts: pl.DataFrame, *, limit: int = 15) -> list[tuple[str, int]]:
    return top_loggers(filter_severity_levels(level_counts, levels=ERROR_PLUS_LEVELS), limit=limit)


def top_hosts_by_errors(level_counts: pl.DataFrame, *, limit: int = 15) -> list[tuple[str, int]]:
    error_counts = filter_severity_levels(level_counts, levels=ERROR_PLUS_LEVELS)
    if error_counts.is_empty():
        return []
    rows = (
        error_counts
        .group_by("hostname")
        .agg(pl.col("log_count").sum().alias("log_count"))
        .sort("log_count", descending=True)
        .head(limit)
        .iter_rows(named=True)
    )
    return [(row["hostname"], int(row["log_count"])) for row in rows]


def top_exception_types(
    exception_counts: pl.DataFrame, *, limit: int = 15
) -> list[tuple[str, int]]:
    if exception_counts.is_empty():
        return []
    rows = (
        exception_counts
        .filter(pl.col("exc_type") != NO_EXC_TYPE_LABEL)
        .group_by("exc_type")
        .agg(pl.col("exception_count").sum().alias("exception_count"))
        .sort("exception_count", descending=True)
        .head(limit)
        .iter_rows(named=True)
    )
    return [(row["exc_type"], int(row["exception_count"])) for row in rows]


def _logger_subsystem(logger_name: str) -> str:
    for subsystem, prefixes in LOGGER_SUBSYSTEMS:
        if any(logger_name.startswith(prefix) for prefix in prefixes):
            return subsystem
    return "Other"


def summarize_by_subsystem(level_counts: pl.DataFrame) -> list[SubsystemSummary]:
    if level_counts.is_empty():
        return []

    totals: dict[str, int] = {}
    error_plus_totals: dict[str, int] = {}
    for row in level_counts.iter_rows(named=True):
        subsystem = _logger_subsystem(row["logger"])
        count = int(row["log_count"])
        totals[subsystem] = totals.get(subsystem, 0) + count
        if row["level"] in ERROR_PLUS_LEVELS:
            error_plus_totals[subsystem] = error_plus_totals.get(subsystem, 0) + count

    summaries = [
        SubsystemSummary(
            subsystem=subsystem,
            total_lines=totals[subsystem],
            error_plus_lines=error_plus_totals.get(subsystem, 0),
        )
        for subsystem in totals
    ]
    return sorted(summaries, key=lambda item: item.error_plus_lines, reverse=True)


def _format_short_date(value: date) -> str:
    return value.strftime("%b %d").replace(" 0", " ")


def describe_filter_scope(
    filters: LogDashboardFilters,
    bounds: tuple[date, date] | None,
    all_environments: list[str],
    all_levels: list[str],
) -> str:
    if filters.environments is not None:
        if len(all_environments) == 1 or set(filters.environments) != set(all_environments):
            env_part = ", ".join(sorted(filters.environments))
        else:
            env_part = "all environments"
    else:
        env_part = "all environments"

    if filters.date_from is not None and filters.date_to is not None:
        date_part = f"{_format_short_date(filters.date_from)}-{_format_short_date(filters.date_to)}"
    elif bounds is not None:
        date_part = f"{_format_short_date(bounds[0])}-{_format_short_date(bounds[1])}"
    else:
        date_part = "all dates"

    if filters.levels is not None and set(filters.levels) != set(all_levels):
        ordered_levels = [level for level in LOG_LEVEL_ORDER if level in filters.levels]
        ordered_levels.extend(
            level for level in sorted(filters.levels) if level not in LOG_LEVEL_ORDER
        )
        level_part = ", ".join(ordered_levels)
    else:
        level_part = "all levels"

    return f"{env_part} · {date_part} · {level_part}"


def format_hours_ago(hours: float | None) -> str:
    if hours is None:
        return "—"
    if hours < 1:
        return "<1h ago"
    return f"{round(hours)}h ago"


def format_stale_data_warning(freshness: DataFreshness) -> str:
    """Human-readable warning when hourly aggregates may be stale."""
    return (
        f"Latest hourly bucket is {format_hours_ago(freshness.hours_since_latest)} "
        f"(>{STALE_BUCKET_THRESHOLD_HOURS}h). Aggregates may be stale — run the maintenance pipeline."
    )


def format_period_delta_detail(delta: PeriodDelta) -> str:
    if delta.previous == 0 and delta.current == 0:
        return "No prior period data"
    sign = "+" if delta.delta >= 0 else ""
    if delta.delta_pct is not None:
        pct_sign = "+" if delta.delta_pct >= 0 else ""
        return f"{sign}{delta.delta} ({pct_sign}{delta.delta_pct:.0f}%) vs prior 24h"
    return f"{sign}{delta.delta} vs prior 24h"
