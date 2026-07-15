from datetime import date

import altair as alt
import streamlit as st

from chess_teacher.other.log_analytics import (
    LOG_LEVEL_COLORS,
    LOG_LEVEL_ORDER,
    SEVERITY_LEVELS,
    LogDashboardFilters,
    apply_exception_filters,
    apply_log_level_filters,
    build_exception_volume_timeseries,
    build_log_volume_timeseries,
    compute_health_metrics,
    compute_log_dashboard_summary,
    default_filter_date_from,
    describe_filter_scope,
    format_hours_ago,
    format_period_delta_detail,
    format_stale_data_warning,
    get_bucket_bounds,
    load_exception_hourly_counts,
    load_log_level_hourly_counts,
    load_recent_warning_error_logs,
    recent_warning_error_log_rows,
    summarize_by_subsystem,
    top_error_loggers,
    top_exception_types,
    top_hosts_by_errors,
)
from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.general_utils import get_current_datetime
from chess_teacher.utils.logging import get_logger
from streamlit_utils.admin_auth import require_admin_user
from streamlit_utils.charts import configure_admin_log_chart, divider_matched_axis
from streamlit_utils.layout import ingest_row_stretch_css
from streamlit_utils.overview_metric import ingest_overview_metrics_css, render_overview_metric
from streamlit_utils.page_config import configure_page
from streamlit_utils.page_logging import log_page_view

configure_page("Admin")
user = require_admin_user()
log_page_view("Admin", user)
logger = get_logger()
db_client = get_db_client()

st.title("Logging dashboard")
st.caption(
    "Hourly aggregates from the maintenance log pipeline (rolling 7-day recompute, Redis-cached)."
)

level_counts = load_log_level_hourly_counts(db_client)
exception_counts = load_exception_hourly_counts(db_client)

if level_counts.is_empty() and exception_counts.is_empty():
    logger.info("Admin logging dashboard empty")
    st.info("No log aggregates yet. Run the maintenance pipeline to populate hourly tables.")
    st.stop()

bounds = get_bucket_bounds(level_counts if not level_counts.is_empty() else exception_counts)
available_environments = sorted(
    set(level_counts["environment"].unique().to_list())
    | set(exception_counts["environment"].unique().to_list())
)
available_levels = sorted(
    level_counts["level"].unique().to_list(),
    key=lambda level: (
        LOG_LEVEL_ORDER.index(level) if level in LOG_LEVEL_ORDER else len(LOG_LEVEL_ORDER)
    ),
)


def _format_count(value: int) -> str:
    return f"{value:,}"


def _build_filters() -> LogDashboardFilters:
    use_date_filter = bounds is not None
    date_from: date | None = None
    date_to: date | None = None
    environment_filter: frozenset[str] | None = None
    level_filter: frozenset[str] | None = None

    with st.expander("Filters", expanded=True):
        if use_date_filter:
            min_date, max_date = bounds
            date_cols = st.columns(2)
            date_from = date_cols[0].date_input(
                "From",
                value=default_filter_date_from(max_date),
                min_value=min_date,
                max_value=max_date,
                key="admin_log_filter_date_from",
            )
            date_to = date_cols[1].date_input(
                "To",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key="admin_log_filter_date_to",
            )
            if date_from > date_to:
                st.warning("Start date is after end date.")
        else:
            st.caption("No dated buckets — date filter unavailable.")

        filter_cols = st.columns(2)
        if len(available_environments) > 1:
            selected_environments = filter_cols[0].multiselect(
                "Environment",
                options=available_environments,
                default=available_environments,
                key="admin_log_filter_environments",
            )
            environment_filter = frozenset(selected_environments)
        elif available_environments:
            environment_filter = frozenset(available_environments)

        if len(available_levels) > 1:
            selected_levels = filter_cols[1].multiselect(
                "Log level",
                options=available_levels,
                default=available_levels,
                key="admin_log_filter_levels",
            )
            level_filter = frozenset(selected_levels)
        elif available_levels:
            level_filter = frozenset(available_levels)

    return LogDashboardFilters(
        date_from=date_from if use_date_filter else None,
        date_to=date_to if use_date_filter else None,
        environments=environment_filter,
        levels=level_filter,
    )


def _render_filter_scope_summary(filters: LogDashboardFilters) -> None:
    st.caption(describe_filter_scope(filters, bounds, available_environments, available_levels))


def _render_stale_warning(freshness) -> None:
    if not freshness.is_stale:
        return
    st.warning(format_stale_data_warning(freshness))


def _render_health_strip(health) -> None:
    ingest_overview_metrics_css()
    ingest_row_stretch_css("admin_log_health")
    freshness_help = (
        health.freshness.latest_bucket.strftime("%Y-%m-%d %H:%M UTC")
        if health.freshness.latest_bucket is not None
        else "No bucket timestamps available"
    )
    with st.container(key="admin_log_health"):
        cols = st.columns(4)
        with cols[0]:
            render_overview_metric(
                "ERROR+ (24h)",
                _format_count(health.error_plus_24h.current),
                details=[format_period_delta_detail(health.error_plus_24h)],
            )
        with cols[1]:
            render_overview_metric(
                "CRITICAL (24h)",
                _format_count(health.critical_24h),
                help="CRITICAL log lines in the last 24 hourly buckets",
            )
        with cols[2]:
            render_overview_metric(
                "Error rate",
                f"{health.error_rate_pct:.1f}%",
                help="ERROR+ lines ÷ all log lines in the last 24h",
            )
        with cols[3]:
            render_overview_metric(
                "Data freshness",
                format_hours_ago(health.freshness.hours_since_latest),
                help=freshness_help,
            )


def _render_stacked_level_chart(
    level_data,
    *,
    title: str,
    levels: frozenset[str] | None = None,
    empty_caption: str,
    use_bars: bool = False,
) -> None:
    st.subheader(title)
    series = build_log_volume_timeseries(level_data, levels=levels)
    if series.is_empty():
        st.caption(empty_caption)
        return

    levels_present = [
        level for level in LOG_LEVEL_ORDER if level in series["level"].unique().to_list()
    ]
    levels_present += [
        level for level in sorted(series["level"].unique().to_list()) if level not in levels_present
    ]
    chart_data = series.to_pandas()
    color_scale = alt.Scale(
        domain=levels_present,
        range=[LOG_LEVEL_COLORS.get(level, "#6b7280") for level in levels_present],
    )
    encode_common = {
        "x": alt.X(
            "bucket_start:T",
            title="Hour (UTC)",
            axis=alt.Axis(format="%a %d %H:%M", labelAngle=-35, labelOverlap=False),
        ),
        "y": alt.Y("log_count:Q", stack="zero", axis=divider_matched_axis(title="Log lines")),
        "color": alt.Color("level:N", scale=color_scale, legend=alt.Legend(title="Level")),
        "tooltip": [
            alt.Tooltip("bucket_start:T", title="Hour", format="%Y-%m-%d %H:%M UTC"),
            alt.Tooltip("level:N", title="Level"),
            alt.Tooltip("log_count:Q", title="Lines", format=","),
        ],
    }

    if use_bars:
        mark = alt.Chart(chart_data).mark_bar(stroke="white", strokeWidth=0.6, opacity=0.92)
    else:
        mark = alt.Chart(chart_data).mark_area(
            interpolate="step-after",
            line={"strokeWidth": 1.5, "color": "white"},
            opacity=0.88,
        )

    chart = configure_admin_log_chart(
        mark.encode(**encode_common),
        height=320,
    )
    st.altair_chart(chart, width="stretch")


def _render_severity_volume_chart(level_data) -> None:
    _render_stacked_level_chart(
        level_data,
        title="WARNING / ERROR / CRITICAL volume",
        levels=SEVERITY_LEVELS,
        empty_caption="No WARNING+ data in the current selection.",
        use_bars=True,
    )


def _render_all_levels_chart(level_data) -> None:
    _render_stacked_level_chart(
        level_data,
        title="Log volume over time",
        levels=None,
        empty_caption="No log level data in the current selection.",
    )


def _render_exception_volume_chart(exception_data) -> None:
    st.subheader("Exception event volume over time")
    series = build_exception_volume_timeseries(exception_data)
    if series.is_empty():
        st.caption("No exception aggregate data in the current selection.")
        return

    chart_data = series.to_pandas()
    chart = configure_admin_log_chart(
        alt
        .Chart(chart_data)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X(
                "bucket_start:T",
                title="Hour (UTC)",
                axis=alt.Axis(format="%a %d %H:%M", labelAngle=-35, labelOverlap=False),
            ),
            y=alt.Y(
                "exception_count:Q",
                axis=divider_matched_axis(title="Events"),
            ),
            tooltip=[
                alt.Tooltip("bucket_start:T", title="Hour", format="%Y-%m-%d %H:%M UTC"),
                alt.Tooltip("exception_count:Q", title="Events", format=","),
            ],
        ),
        height=240,
    )
    st.altair_chart(chart, width="stretch")


def _render_recent_warning_error_logs(filters: LogDashboardFilters) -> None:
    st.subheader("Latest WARNING / ERROR / CRITICAL messages")
    recent_logs = load_recent_warning_error_logs(db_client, filters)
    rows = recent_warning_error_log_rows(recent_logs)
    if not rows:
        st.caption("No warning or error messages in the current selection.")
        return
    st.dataframe(rows, hide_index=True, width="stretch")


def _render_error_tables(level_data, exception_data) -> None:
    table_cols = st.columns(3)

    with table_cols[0]:
        st.subheader("Top error loggers")
        logger_rows = top_error_loggers(level_data)
        if not logger_rows:
            st.caption("No ERROR+ logger data in the current selection.")
        else:
            st.dataframe(
                [{"Logger": name, "Lines": count} for name, count in logger_rows],
                hide_index=True,
                width="stretch",
            )

    with table_cols[1]:
        st.subheader("Top exception types")
        exception_rows = top_exception_types(exception_data)
        if not exception_rows:
            st.caption("No traced exceptions in the current selection.")
        else:
            st.dataframe(
                [{"Exception": exc_type, "Count": count} for exc_type, count in exception_rows],
                hide_index=True,
                width="stretch",
            )

    with table_cols[2]:
        st.subheader("Top hosts (ERROR+)")
        host_rows = top_hosts_by_errors(level_data)
        if not host_rows:
            st.caption("No ERROR+ host data in the current selection.")
        else:
            st.dataframe(
                [{"Host": host, "Errors": count} for host, count in host_rows],
                hide_index=True,
                width="stretch",
            )


def _render_subsystem_table(level_data) -> None:
    rows = summarize_by_subsystem(level_data)
    if not rows:
        st.caption("No logger data in the current selection.")
        return
    st.dataframe(
        [
            {
                "Subsystem": row.subsystem,
                "Total lines": row.total_lines,
                "ERROR+ lines": row.error_plus_lines,
            }
            for row in rows
        ],
        hide_index=True,
        width="stretch",
    )


def _render_volume_cardinality(summary) -> None:
    ingest_overview_metrics_css()
    ingest_row_stretch_css("admin_log_cardinality")
    with st.container(key="admin_log_cardinality"):
        cols = st.columns(4)
        with cols[0]:
            render_overview_metric("Distinct loggers", _format_count(summary.distinct_loggers))
        with cols[1]:
            render_overview_metric("Distinct hosts", _format_count(summary.distinct_hosts))
        with cols[2]:
            render_overview_metric(
                "WARNING+ events",
                _format_count(summary.warning_error_events),
                help="Rows in exception_hourly_counts (includes logs without a traceback).",
            )
        with cols[3]:
            render_overview_metric(
                "Traced exceptions",
                _format_count(summary.traced_exceptions),
                help="WARNING+ rows with a real exc_type.",
            )


filters = _build_filters()
filtered_levels = apply_log_level_filters(level_counts, filters)
filtered_exceptions = apply_exception_filters(exception_counts, filters)

reference_now = get_current_datetime()
health = compute_health_metrics(
    level_counts,
    exception_counts,
    reference_now=reference_now,
    environments=filters.environments,
)

_render_filter_scope_summary(filters)
_render_stale_warning(health.freshness)
_render_health_strip(health)

if filtered_levels.is_empty() and filtered_exceptions.is_empty():
    st.warning("No log aggregates match the current filters.")
    st.stop()

summary = compute_log_dashboard_summary(filtered_levels, filtered_exceptions)
st.divider()
_render_severity_volume_chart(filtered_levels)
st.divider()
_render_recent_warning_error_logs(filters)
st.divider()
_render_error_tables(filtered_levels, filtered_exceptions)

with st.expander("Subsystem breakdown", expanded=False):
    _render_subsystem_table(filtered_levels)

with st.expander("All log levels", expanded=False):
    _render_all_levels_chart(filtered_levels)

with st.expander("Exception event volume", expanded=False):
    _render_exception_volume_chart(filtered_exceptions)

with st.expander("Volume & cardinality", expanded=False):
    _render_volume_cardinality(summary)
