from contextlib import nullcontext
from datetime import date

import altair as alt
import polars as pl
import streamlit as st

from chess_teacher.other.game_statistics import (
    RESULT_LABELS,
    TIME_CONTROL_CLASSES,
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
from chess_teacher.platform.users_accounts import get_accounts_for_user
from chess_teacher.utils.chess_utils import Color, Result
from chess_teacher.utils.db_client import get_db_client
from streamlit_utils.chart_legend import rating_legend_items, render_series_legend, series_colors
from streamlit_utils.layout import column_with_divider
from streamlit_utils.login import require_authenticated_user
from streamlit_utils.pie_chart import (
    account_pie_slices,
    render_pie_chart,
    render_pie_charts_row,
    result_pie_slices,
)
from streamlit_utils.platform_ui import pick_accounts_multi

_RESULT_OPTIONS = list(Result)
_RESULT_CHART_COLORS = {
    RESULT_LABELS[Result.WIN.value]: "#22c55e",
    RESULT_LABELS[Result.DRAW.value]: "#9ca3af",
    RESULT_LABELS[Result.LOSS.value]: "#ef4444",
    RESULT_LABELS[Result.NO_RESULT.value]: "#eab308",
}
_COLOR_OPTIONS = list(Color)
user = require_authenticated_user()
db_client = get_db_client()
accounts = get_accounts_for_user(user, db_client)

st.title("Game statistics")
st.caption("Summary from ingested games across your linked platform accounts.")

if not accounts:
    st.info("Link a platform account in **Settings**, then run the **Pipeline** to load games.")
    st.stop()

accounts_by_id = {account.account_id: account for account in accounts}
all_account_ids = [account.account_id for account in accounts]

games = load_games_for_accounts(db_client, all_account_ids)

if games.is_empty():
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


def _format_date(value) -> str:
    if value is None:
        return "—"
    return value.strftime("%Y-%m-%d")


def _format_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.1f}%"


def _render_overview(stats: GameStatisticsSummary) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Games played", stats.total_games)
    col2.metric("Win rate", _format_pct(stats.win_rate_pct))
    col3.metric("Average rating", int(stats.avg_user_elo) if stats.avg_user_elo else "—")
    col4.metric(
        "Date range",
        f"{_format_date(stats.earliest_game)} → {_format_date(stats.latest_game)}",
    )

    record_cols = st.columns(4)
    record_cols[0].metric("Wins", stats.wins)
    record_cols[1].metric("Draws", stats.draws)
    record_cols[2].metric("Losses", stats.losses)
    record_cols[3].metric("No result", stats.no_result)


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

    chart = (
        alt
        .Chart(chart_data)
        .mark_line(strokeWidth=1, point=False)
        .encode(
            x=alt.X("start_time:T", title="Game date"),
            y=alt.Y("user_elo:Q", title="Your rating", scale=alt.Scale(zero=False)),
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
    )
    st.altair_chart(chart, use_container_width=True)
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
    account_rows = (
        [{"account": account, "games": count} for account, count in stats.games_by_account.items()]
        if show_account
        else []
    )

    if show_account and account_rows:
        render_pie_charts_row([
            ("Results", result_pie_slices(result_rows, result_colors)),
            ("Games by account", account_pie_slices(account_rows, accounts_by_id)),
        ])
    else:
        render_pie_chart("Results", result_pie_slices(result_rows, result_colors))


def _render_color_breakdown(
    stats: GameStatisticsSummary,
    *,
    colors: tuple[str, ...],
) -> None:
    st.subheader("Performance by color")
    if not colors:
        return

    cols = st.columns(len(colors), gap="small")
    for index, color in enumerate(colors):
        breakdown = stats.color_breakdown[color]
        with cols[index]:
            wrapper = (
                column_with_divider(f"color_{index}") if index < len(colors) - 1 else nullcontext()
            )
            with wrapper:
                st.markdown(f"**{color.title()}**")
                st.metric("Games", breakdown.games)
                st.metric("Win rate", _format_pct(breakdown.win_rate_pct))


def _render_openings(stats: GameStatisticsSummary) -> None:
    st.subheader("Top openings (ECO)")
    if not stats.top_openings:
        st.caption("No ECO codes in loaded games yet.")
        return
    st.dataframe(
        [{"ECO": eco, "Games": count} for eco, count in stats.top_openings],
        hide_index=True,
        use_container_width=True,
    )


filters = _build_filters()
filtered_games = apply_filters(games, filters)

if filtered_games.is_empty():
    st.warning("No games match the current filters.")
    st.stop()

filtered_count = filtered_games.height
if filtered_count < total_games_loaded:
    st.caption(f"Showing **{filtered_count}** of **{total_games_loaded}** loaded games.")
if undated_games and (filters.date_from is not None or filters.date_to is not None):
    st.caption(f"{undated_games} game(s) without a start time are excluded by the date filter.")

color_breakdown_colors = tuple(filters.colors or (Color.WHITE.value, Color.BLACK.value))
summary = compute_summary(
    filtered_games,
    accounts_by_id,
    color_breakdown_colors=color_breakdown_colors,
)
_show_account_breakdown = len(accounts) > 1 and len(filters.account_ids or ()) > 1

_render_overview(summary)
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
