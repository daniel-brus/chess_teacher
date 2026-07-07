from datetime import date, datetime

import altair as alt
import polars as pl
import streamlit as st

from chess_teacher.other.game_statistics import (
    RESULT_LABELS,
    TIME_CONTROL_CLASSES,
    ColorBreakdown,
    GameFilters,
    GameStatisticsSummary,
    apply_filters,
    build_rating_history,
    compute_summary,
    get_dated_bounds,
    load_games_for_accounts,
    sorted_time_controls,
    with_time_control_class,
)
from chess_teacher.platform.account import Account
from chess_teacher.utils.chess_utils import Color, Result
from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.logging import get_logger
from streamlit_utils.charts import (
    account_category_pie_slices,
    apply_divider_matched_axis_config,
    divider_matched_axis,
    ensure_vega_chart_transparent_css,
    rating_chart_x_axis,
    rating_legend_items,
    render_pie_chart,
    render_pie_charts_row,
    render_series_legend,
    result_pie_slices,
    series_colors,
)
from streamlit_utils.layout import (
    column_with_divider,
    ingest_column_divider_css,
    ingest_row_stretch_css,
)
from streamlit_utils.login import require_authenticated_user
from streamlit_utils.overview_metric import (
    ingest_overview_metrics_css,
    render_overview_metric,
    render_overview_section_title,
)
from streamlit_utils.page_config import configure_page
from streamlit_utils.page_logging import log_page_view
from streamlit_utils.platform_ui import pick_accounts_multi

configure_page("Statistics")

logger = get_logger()
_RESULT_OPTIONS = list(Result)
_RESULT_CHART_COLORS = {
    RESULT_LABELS[Result.WIN.value]: "#22c55e",
    RESULT_LABELS[Result.DRAW.value]: "#9ca3af",
    RESULT_LABELS[Result.LOSS.value]: "#ef4444",
    RESULT_LABELS[Result.NO_RESULT.value]: "#eab308",
}
_COLOR_OPTIONS = list(Color)
user = require_authenticated_user()
log_page_view("Statistics", user)
db_client = get_db_client()
accounts = user.get_linked_accounts(db_client)

st.title("Game statistics")
st.caption("Summary from ingested games across your linked platform accounts.")

if not accounts:
    logger.info("Statistics page empty: no linked accounts user_id=%s", user.user_id)
    st.info("Link a platform account in **Settings**, then run the **Pipeline** to load games.")
    st.stop()

accounts_by_id = {account.account_id: account for account in accounts}
all_account_ids = [account.account_id for account in accounts]

games = load_games_for_accounts(db_client, all_account_ids)

if games.is_empty():
    logger.info(
        "Statistics page empty: no games loaded user_id=%s account_count=%s",
        user.user_id,
        len(all_account_ids),
    )
    st.info("No games found yet. Run the pipeline on a linked account to ingest games.")
    st.stop()

total_games_loaded = games.height
available_variants = sorted(games["variant"].drop_nulls().unique().to_list())
games_with_time_control = with_time_control_class(games)
available_time_controls = sorted_time_controls(
    games_with_time_control["time_control"].unique().to_list()
)
dated_bounds = get_dated_bounds(games)
undated_games = games.filter(pl.col("start_time").is_null()).height


def _build_filters() -> GameFilters:
    account_filter: frozenset[str]
    with st.expander("Filters", expanded=False):
        if len(accounts) > 1:
            selected_accounts = pick_accounts_multi(
                accounts,
                key_prefix="stats_filter_account",
            )
            account_filter = frozenset(account.account_id for account in selected_accounts)
        else:
            account_filter = frozenset({accounts[0].account_id})

        use_date_filter = dated_bounds is not None
        date_from: date | None = None
        date_to: date | None = None

        if use_date_filter:
            min_date, max_date = dated_bounds
            date_cols = st.columns(2)
            date_from = date_cols[0].date_input(
                "From",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                key="stats_filter_date_from",
            )
            date_to = date_cols[1].date_input(
                "To",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key="stats_filter_date_to",
            )
            if date_from > date_to:
                st.warning("Start date is after end date.")
        else:
            st.caption("No dated games — date filter unavailable.")

        filter_cols = st.columns(2)
        selected_colors = filter_cols[0].multiselect(
            "Color",
            options=_COLOR_OPTIONS,
            default=_COLOR_OPTIONS,
            format_func=lambda color: color.value.title(),
            key="stats_filter_colors",
        )
        selected_results = filter_cols[1].multiselect(
            "Result",
            options=_RESULT_OPTIONS,
            default=_RESULT_OPTIONS,
            format_func=lambda result: RESULT_LABELS[result.value],
            key="stats_filter_results",
        )

        variant_filter: frozenset[str] | None = None
        if len(available_variants) > 1:
            default_variants = ["standard"] if "standard" in available_variants else []
            selected_variants = st.multiselect(
                "Variant",
                options=available_variants,
                default=default_variants,
                key="stats_filter_variants",
            )
            variant_filter = frozenset(selected_variants)
        elif available_variants == ["standard"]:
            variant_filter = frozenset({"standard"})

        default_time_controls = [
            tc for tc in TIME_CONTROL_CLASSES if tc in available_time_controls and tc != "Unknown"
        ]
        selected_time_controls = st.multiselect(
            "Time control",
            options=available_time_controls,
            default=default_time_controls,
            help="Estimated from initial + 40*increment: UltraBullet <30s, Bullet <3m, Blitz <10m, Rapid ≥10m.",
            key="stats_filter_time_controls",
        )

    return GameFilters(
        date_from=date_from if use_date_filter else None,
        date_to=date_to if use_date_filter else None,
        colors=frozenset(color.value for color in selected_colors),
        results=frozenset(result.value for result in selected_results),
        variants=variant_filter,
        account_ids=account_filter,
        time_controls=frozenset(selected_time_controls),
    )


def _format_date_dmy(value: datetime | date | None) -> str:
    if value is None:
        return "—"
    if isinstance(value, datetime):
        value = value.date()
    return value.strftime("%d/%m/%Y")


def _format_time_ago(value: datetime | date | None, *, today: date | None = None) -> str:
    if value is None:
        return "—"
    if isinstance(value, datetime):
        value = value.date()
    today = today or date.today()
    days = max(0, (today - value).days)
    if days == 0:
        return "today"
    if days < 30:
        unit = "day" if days == 1 else "days"
        return f"{days} {unit} ago"
    if days < 365:
        months = max(1, days // 30)
        unit = "month" if months == 1 else "months"
        return f"{months} {unit} ago"
    years = days // 365
    months = (days % 365) // 30
    if months >= 1:
        year_unit = "year" if years == 1 else "years"
        month_unit = "month" if months == 1 else "months"
        return f"{years} {year_unit}, {months} {month_unit} ago"
    unit = "year" if years == 1 else "years"
    return f"{years} {unit} ago"


def _format_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.1f}%"


def _format_count(value: int) -> str:
    return f"{value:,}"


def _render_overview(
    stats: GameStatisticsSummary,
    *,
    filter_caption: str | None = None,
) -> None:
    ingest_overview_metrics_css()
    ingest_row_stretch_css("stats_overview")
    with st.container(key="stats_overview"):
        if filter_caption:
            with st.columns(1)[0]:
                st.caption(filter_caption)
        row1 = st.columns(4)
        row2 = st.columns(4)

        with row1[0]:
            render_overview_metric(
                "Games played (rated)",
                f"{_format_count(stats.total_games)} ({_format_count(stats.rated_games)})",
            )

        with row1[1]:
            render_overview_metric(
                "Win rate",
                _format_pct(stats.win_rate_pct),
                help="Decisive results only (wins, draws, and losses).",
            )

        with row1[2]:
            render_overview_metric(
                "Record (W - D - L)",
                f"{_format_count(stats.wins)} - {_format_count(stats.draws)} - {_format_count(stats.losses)}",
            )

        favorite = stats.favorite_time_control
        with row1[3]:
            if favorite is None:
                render_overview_metric("Favorite time control", "—")
            else:
                render_overview_metric(
                    "Favorite time control",
                    f"{favorite.time_control} ({favorite.share_pct:.1f}% of games)",
                )

        peak = stats.peak_rating
        with row2[0]:
            if peak is None:
                render_overview_metric("Peak rating", "—")
            else:
                render_overview_metric(
                    "Peak rating",
                    f"{peak.rating} ({_format_date_dmy(peak.game_date)})",
                    details=(f"{peak.account_label} · {peak.time_control}",),
                )

        streak = stats.longest_win_streak
        with row2[1]:
            if streak.length == 0:
                render_overview_metric("Longest win streak", "—")
            else:
                render_overview_metric(
                    "Longest win streak",
                    str(streak.length),
                    details=(
                        f"{_format_date_dmy(streak.start_date)} - {_format_date_dmy(streak.end_date)}",
                    ),
                )

        best_win = stats.highest_opponent_beat
        with row2[2]:
            if best_win is None:
                render_overview_metric("Highest opponent beat", "—")
            else:
                render_overview_metric(
                    "Highest opponent beat",
                    str(best_win.opponent_elo),
                    details=(_format_date_dmy(best_win.game_date),),
                )

        with row2[3]:
            if stats.first_game is None:
                render_overview_metric("First game", "—")
            else:
                render_overview_metric(
                    "First game",
                    _format_date_dmy(stats.first_game),
                    details=(_format_time_ago(stats.first_game),),
                )


def _render_rating_chart(
    games: pl.DataFrame,
    accounts_by_id: dict[str, Account],
) -> None:
    st.subheader("Rating over time")
    st.caption("One line per account and time control (Bullet, Blitz, Rapid, …).")
    history = build_rating_history(games, accounts_by_id)
    if history.is_empty():
        st.caption("No games with both a start time and rating in the current selection.")
        return

    series_ids = sorted(history["series_id"].unique().to_list())
    color_scale = series_colors(series_ids)
    chart_data = history.to_pandas()
    date_min = chart_data["start_time"].min()
    date_max = chart_data["start_time"].max()

    chart = apply_divider_matched_axis_config(
        alt
        .Chart(chart_data)
        .mark_line(strokeWidth=1, point=False)
        .encode(
            x=alt.X(
                "start_time:T",
                axis=rating_chart_x_axis(date_min, date_max, title="Game date"),
            ),
            y=alt.Y(
                "user_elo:Q",
                scale=alt.Scale(zero=False),
                axis=divider_matched_axis(title="Your rating"),
            ),
            color=alt.Color(
                "series_id:N",
                scale=alt.Scale(
                    domain=series_ids,
                    range=[color_scale[series_id] for series_id in series_ids],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("start_time:T", title="Date"),
                alt.Tooltip("user_elo:Q", title="Rating"),
                alt.Tooltip("series_label:N", title="Series"),
                alt.Tooltip("account:N", title="Account"),
                alt.Tooltip("time_control:N", title="Time control"),
            ],
        )
        .configure_line(strokeWidth=1, point=False)
        .configure(background="transparent")
        .configure_view(fill=None, strokeWidth=0)
    )
    ensure_vega_chart_transparent_css()
    st.altair_chart(chart, width="stretch")
    render_series_legend(rating_legend_items(history, accounts_by_id))


def _result_chart_colors(result_rows: list[dict[str, object]]) -> dict[str, str]:
    labels = {row["result"] for row in result_rows}
    return {label: color for label, color in _RESULT_CHART_COLORS.items() if label in labels}


def _render_distribution_charts(
    stats: GameStatisticsSummary,
    accounts_by_id: dict[str, Account],
    *,
    show_account: bool,
) -> None:
    result_rows = [
        {"result": label, "games": count}
        for label, count in stats.result_counts.items()
        if count > 0
    ]
    result_colors = _result_chart_colors(result_rows)
    account_category_counts = stats.games_by_account_and_category if show_account else ()

    if show_account and account_category_counts:
        render_pie_charts_row([
            ("Results", result_pie_slices(result_rows, result_colors)),
            (
                "Games by account and category",
                account_category_pie_slices(account_category_counts, accounts_by_id),
            ),
        ])
    else:
        render_pie_chart("Results", result_pie_slices(result_rows, result_colors))


_DECISIVE_RESULTS_HELP = "Decisive results only (wins, draws, and losses)."


def _render_color_column(breakdown: ColorBreakdown) -> None:
    render_overview_metric(
        "Games played (rated)",
        f"{_format_count(breakdown.games)} ({_format_count(breakdown.rated_games)})",
    )
    render_overview_metric(
        "Record (W - D - L)",
        (
            f"{_format_count(breakdown.wins)} - {_format_count(breakdown.draws)} - "
            f"{_format_count(breakdown.losses)}"
        ),
    )
    render_overview_metric(
        "Win rate",
        _format_pct(breakdown.win_rate_pct),
        help=_DECISIVE_RESULTS_HELP,
    )
    render_overview_metric(
        "Loss rate",
        _format_pct(breakdown.loss_rate_pct),
        help=_DECISIVE_RESULTS_HELP,
    )
    if breakdown.favorite_opening is None:
        render_overview_metric("Favourite opening", "—")
    else:
        opening_details: tuple[str, ...] = ()
        if breakdown.favorite_opening_share_pct is not None:
            opening_details = (f"{breakdown.favorite_opening_share_pct:.1f}% of games",)
        render_overview_metric(
            "Favourite opening",
            breakdown.favorite_opening,
            details=opening_details,
        )
    streak = breakdown.longest_win_streak
    if streak.length == 0:
        render_overview_metric("Longest win streak", "—")
    else:
        render_overview_metric(
            "Longest win streak",
            str(streak.length),
            details=(
                f"{_format_date_dmy(streak.start_date)} - {_format_date_dmy(streak.end_date)}",
            ),
        )
    best_win = breakdown.best_opponent_beat
    if best_win is None:
        render_overview_metric("Best opponent beat", "—")
    else:
        render_overview_metric(
            "Best opponent beat",
            str(best_win.opponent_elo),
            details=(_format_date_dmy(best_win.game_date),),
        )


def _render_color_breakdown(
    stats: GameStatisticsSummary,
    *,
    colors: tuple[str, ...],
) -> None:
    st.subheader("Performance by color")
    if not colors:
        return

    ingest_overview_metrics_css()
    ingest_row_stretch_css("color_performance")
    with st.container(key="color_performance"):
        for index in range(len(colors) - 1):
            ingest_column_divider_css(f"color_col_{index}")
        cols = st.columns(len(colors), gap="small")
        for index, color in enumerate(colors):
            breakdown = stats.color_breakdown[color]
            with cols[index]:
                with column_with_divider(f"color_col_{index}"):
                    render_overview_section_title(color.title())
                    _render_color_column(breakdown)


def _render_openings(stats: GameStatisticsSummary) -> None:
    st.subheader("Top opening families")
    if not stats.top_openings:
        st.caption("No opening families in loaded games yet.")
        return
    st.dataframe(
        [{"Opening family": family, "Games": count} for family, count in stats.top_openings],
        hide_index=True,
        width="stretch",
    )


filters = _build_filters()
filtered_games = apply_filters(games, filters)

if filtered_games.is_empty():
    st.warning("No games match the current filters.")
    st.stop()

filtered_count = filtered_games.height
filter_caption_parts: list[str] = []
if filtered_count < total_games_loaded:
    filter_caption_parts.append(
        f"Showing **{filtered_count}** of **{total_games_loaded}** loaded games."
    )
if undated_games and (filters.date_from is not None or filters.date_to is not None):
    filter_caption_parts.append(
        f"{undated_games} game(s) without a start time are excluded by the date filter."
    )

color_breakdown_colors = tuple(filters.colors or (Color.WHITE.value, Color.BLACK.value))
summary = compute_summary(
    filtered_games,
    accounts_by_id,
    color_breakdown_colors=color_breakdown_colors,
)
_show_account_breakdown = len(accounts) > 1 and len(filters.account_ids or ()) > 1

_render_overview(
    summary,
    filter_caption=" ".join(filter_caption_parts) if filter_caption_parts else None,
)
st.divider()
_render_rating_chart(filtered_games, accounts_by_id)
st.divider()
_render_distribution_charts(
    summary,
    accounts_by_id,
    show_account=_show_account_breakdown,
)
st.divider()
_render_color_breakdown(summary, colors=color_breakdown_colors)
st.divider()
_render_openings(summary)
