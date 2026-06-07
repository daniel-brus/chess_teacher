"""Streamlit chart helpers: Altair pies and HTML legends (line/swatch markers)."""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

import altair as alt
import pandas as pd
import polars as pl
import streamlit as st

from chess_teacher.other.game_statistics import AccountCategoryGameCount
from chess_teacher.platform.account import Account, AccountPlatform
from chess_teacher.platform.raw_assets import storage_image_data_uri
from streamlit_utils.layout import ingest_css, render_center_divider_column, three_column_row
from streamlit_utils.theme import active_appearance, divider_rgba, divider_width_px

# Vega category10 — keep in sync with rating chart series colors.
_CATEGORY10 = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)

_CHART_SIZE = 168
_LEGEND_MIN_HEIGHT = 48
_TITLE_AND_GAPS = 72
_ROW_WIDTHS = (12, 1, 12)
_ROW_KEY = "charts_pie_row"
_VEGA_CHART_CSS_APPLIED = "_vega_chart_transparent_css"

LegendMarker = Literal["line", "swatch"]


def ensure_vega_chart_transparent_css() -> None:
    """Once per session: Vega-Lite embed wrapper matches page background."""
    if st.session_state.get(_VEGA_CHART_CSS_APPLIED):
        return
    st.session_state[_VEGA_CHART_CSS_APPLIED] = True
    ingest_css(
        """
<style>
[data-testid="stVegaLiteChart"],
[data-testid="stVegaLiteChart"] > div,
[data-testid="stVegaLiteChart"] .vega-embed,
[data-testid="stVegaLiteChart"] details {
    background: transparent !important;
    background-color: transparent !important;
}
</style>
"""
    )


@dataclass(frozen=True)
class SeriesLegendItem:
    series_id: str
    color: str
    label: str
    platform: AccountPlatform | None
    logo_key: str | None
    share_pct: float | None = None


@dataclass(frozen=True)
class PieChartSlice:
    label: str
    value: float
    color: str
    logo_key: str | None = None
    platform: AccountPlatform | None = None


def _divider_axis_kwargs(*, grid: bool = True) -> dict[str, object]:
    color = divider_rgba()
    width = divider_width_px()
    return {
        "grid": grid,
        "gridColor": color,
        "gridOpacity": 1,
        "gridWidth": width,
        "domainColor": color,
        "domainOpacity": 1,
        "domainWidth": width,
        "tickColor": color,
        "tickOpacity": 1,
        "tickWidth": width,
    }


def divider_matched_axis(**kwargs: object) -> alt.Axis:
    """Grid, domain, and ticks aligned with app dividers (beats Streamlit Vega theme)."""
    grid = bool(kwargs.pop("grid", True))
    return alt.Axis(**{**_divider_axis_kwargs(grid=grid), **kwargs})


def apply_divider_matched_axis_config(chart: alt.Chart) -> alt.Chart:
    """Rating chart axes: horizontal grids on Y only; X keeps ticks, no vertical grid."""
    y_kw = _divider_axis_kwargs()
    x_kw = {**_divider_axis_kwargs(), "grid": False}
    return chart.configure_axisXTemporal(**x_kw).configure_axisYQuantitative(**y_kw)


def _coerce_timestamp(value: datetime | date) -> pd.Timestamp:
    return pd.Timestamp(value)


def _temporal_axis_tick_literal(ts: pd.Timestamp) -> str:
    """ISO date strings Vega-Lite accepts for temporal ``axis.values``."""
    if ts.tz is not None:
        return ts.isoformat()
    return ts.strftime("%Y-%m-%d")


def _month_start_tick_literals(min_ts: pd.Timestamp, max_ts: pd.Timestamp) -> list[str]:
    start = pd.Timestamp(year=min_ts.year, month=min_ts.month, day=1, tz=min_ts.tz)
    end = pd.Timestamp(year=max_ts.year, month=max_ts.month, day=1, tz=max_ts.tz)
    return [_temporal_axis_tick_literal(t) for t in pd.date_range(start, end, freq="MS")]


def _year_start_tick_literals(min_ts: pd.Timestamp, max_ts: pd.Timestamp) -> list[str]:
    literals: list[str] = []
    for year in range(min_ts.year, max_ts.year + 1):
        jan_first = pd.Timestamp(year=year, month=1, day=1, tz=min_ts.tz)
        end_of_year = pd.Timestamp(
            year=year,
            month=12,
            day=31,
            hour=23,
            minute=59,
            second=59,
            tz=min_ts.tz,
        )
        if min_ts <= end_of_year and max_ts >= jan_first:
            literals.append(_temporal_axis_tick_literal(jan_first))
    return literals


def rating_chart_x_axis(
    min_time: datetime | date,
    max_time: datetime | date,
    *,
    title: str = "Game date",
    **kwargs: object,
) -> alt.Axis:
    """X-axis ticks: Jan 1 per overlapping year when span ≥2 years, else month starts.

    If no ticks are produced, Vega chooses ticks automatically with ``%Y/%m/%d`` labels.
    """
    min_ts = _coerce_timestamp(min_time)
    max_ts = _coerce_timestamp(max_time)
    if max_ts < min_ts:
        min_ts, max_ts = max_ts, min_ts

    year_span = max_ts.year - min_ts.year
    if year_span >= 2:
        values = _year_start_tick_literals(min_ts, max_ts)
        tick_format = "%Y"
    else:
        values = _month_start_tick_literals(min_ts, max_ts)
        tick_format = "%b %Y"

    axis_kw: dict[str, object] = {
        "formatType": "time",
        "labelOverlap": False,
        "title": title,
    }
    if not values:
        return divider_matched_axis(format="%Y/%m/%d", grid=False, **axis_kw, **kwargs)

    return divider_matched_axis(
        values=values,
        format=tick_format,
        grid=False,
        **axis_kw,
        **kwargs,
    )


def series_colors(series_ids: list[str]) -> dict[str, str]:
    """Map each ``series_id`` to a hex color (sorted domain matches Altair)."""
    ordered = sorted(series_ids)
    return {
        series_id: _CATEGORY10[index % len(_CATEGORY10)] for index, series_id in enumerate(ordered)
    }


def rating_legend_items(
    history: pl.DataFrame,
    accounts_by_id: dict[str, Account],
) -> list[SeriesLegendItem]:
    """One legend row per ``series_id`` in rating history."""
    meta = (
        history
        .select("series_id", "account_id", "time_control", "series_label")
        .unique(subset=["series_id"])
        .sort("series_id")
    )
    colors = series_colors(meta["series_id"].to_list())
    appearance = active_appearance()
    items: list[SeriesLegendItem] = []
    for row in meta.iter_rows(named=True):
        account = accounts_by_id.get(row["account_id"])
        platform = account.platform if account else None
        logo_key = platform.logo_key(appearance=appearance) if platform else None
        items.append(
            SeriesLegendItem(
                series_id=row["series_id"],
                color=colors[row["series_id"]],
                label=row["series_label"],
                platform=platform,
                logo_key=logo_key,
            )
        )
    return items


def _share_pct_decimals(share_pcts: list[float]) -> int:
    """One precision for every slice in a pie (e.g. all ``.1f`` if any needs it)."""
    if not share_pcts:
        return 0
    if any(p < 10 or abs(p - round(p)) > 0.05 for p in share_pcts):
        return 1
    return 0


def _format_share_pct(share_pct: float, decimals: int) -> str:
    return f"{share_pct:.{decimals}f}%"


def _vega_share_format(decimals: int) -> str:
    return f".{decimals}%" if decimals else ".0%"


def pie_legend_items(slices: list[PieChartSlice]) -> list[SeriesLegendItem]:
    """Legend rows for a pie chart, including share of total games per slice."""
    total = sum(s.value for s in slices)
    return [
        SeriesLegendItem(
            series_id=s.label,
            color=s.color,
            label=s.label,
            platform=s.platform,
            logo_key=s.logo_key,
            share_pct=(100.0 * s.value / total) if total > 0 else None,
        )
        for s in sorted(slices, key=lambda s: s.label)
    ]


def render_series_legend(
    items: list[SeriesLegendItem],
    *,
    marker: LegendMarker = "line",
    align: Literal["left", "center"] = "left",
    min_height_px: int | None = None,
) -> None:
    """Legend below a chart: color marker + optional platform SVG + label."""
    if not items:
        return

    share_pcts = [item.share_pct for item in items if item.share_pct is not None]
    share_decimals = _share_pct_decimals(share_pcts)

    parts: list[str] = []
    icon_size = 14 if marker == "swatch" else 16
    for item in items:
        icon_html = ""
        if item.logo_key is not None:
            data_uri = storage_image_data_uri(item.logo_key)
            if data_uri is not None:
                alt_text = item.platform.value if item.platform else "Platform"
                icon_html = (
                    f'<img src="{data_uri}" width="{icon_size}" height="{icon_size}" '
                    f'alt="{html.escape(alt_text)}" '
                    f'style="vertical-align:middle;margin-right:4px;">'
                )
        if marker == "swatch":
            color_marker = (
                f'<span style="display:inline-block;width:10px;height:10px;'
                f"background:{html.escape(item.color)};"
                f'margin-right:5px;border-radius:2px;"></span>'
            )
        else:
            color_marker = (
                f'<span style="display:inline-block;width:14px;height:3px;'
                f"background:{html.escape(item.color)};"
                f'margin-right:6px;border-radius:1px;"></span>'
            )
        share_html = ""
        if item.share_pct is not None:
            share_html = (
                f' <span style="opacity:0.72;">'
                f"({html.escape(_format_share_pct(item.share_pct, share_decimals))})</span>"
            )
        parts.append(
            "<span style="
            '"display:inline-flex;align-items:center;margin:2px 10px 2px 0;">'
            f"{color_marker}{icon_html}"
            f"<span>{html.escape(item.label)}{share_html}</span></span>"
        )

    font_size = "0.82rem" if marker == "swatch" else "0.9rem"
    justify = "center" if align == "center" else "flex-start"
    min_height = f"min-height:{min_height_px}px;" if min_height_px else ""
    st.html(
        f'<div style="display:flex;flex-wrap:wrap;align-items:center;'
        f'justify-content:{justify};font-size:{font_size};width:100%;{min_height}">'
        + "".join(parts)
        + "</div>"
    )


def result_pie_slices(
    rows: list[dict[str, object]],
    colors: dict[str, str],
) -> list[PieChartSlice]:
    return [
        PieChartSlice(
            label=str(row["result"]),
            value=float(row["games"]),  # type: ignore[arg-type]
            color=colors[str(row["result"])],
        )
        for row in rows
    ]


def account_category_pie_slices(
    counts: Sequence[AccountCategoryGameCount],
    accounts_by_id: dict[str, Account],
) -> list[PieChartSlice]:
    """Pie slices per account and time control (same series keys as the rating chart)."""
    if not counts:
        return []
    colors = series_colors([row.series_id for row in counts])
    appearance = active_appearance()
    return [
        PieChartSlice(
            label=row.series_label,
            value=float(row.games),
            color=colors[row.series_id],
            platform=(
                account.platform if (account := accounts_by_id.get(row.account_id)) else None
            ),
            logo_key=(account.platform.logo_key(appearance=appearance) if account else None),
        )
        for row in sorted(counts, key=lambda row: row.series_label)
    ]


def _pie_chart(slices: list[PieChartSlice], *, size: int) -> alt.Chart:
    domain = [s.label for s in slices]
    color_range = [s.color for s in slices]
    total = sum(s.value for s in slices)
    share_pcts = [(100.0 * s.value / total) if total > 0 else 0.0 for s in slices]
    share_decimals = _share_pct_decimals(share_pcts)
    data = [
        {
            "category": s.label,
            "value": s.value,
            "percent": (s.value / total) if total > 0 else 0.0,
        }
        for s in slices
    ]
    return (
        alt
        .Chart(alt.Data(values=data))
        .mark_arc()
        .encode(
            theta=alt.Theta("value:Q", stack=True),
            color=alt.Color(
                "category:N",
                scale=alt.Scale(domain=domain, range=color_range),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("category:N", title="Category"),
                alt.Tooltip("value:Q", title="Games"),
                alt.Tooltip(
                    "percent:Q",
                    title="Share",
                    format=_vega_share_format(share_decimals),
                ),
            ],
        )
        .properties(width=size, height=size, autosize={"type": "none"})
        .configure(background="transparent")
        .configure_view(fill=None, strokeWidth=0)
    )


def render_pie_chart(title: str, slices: list[PieChartSlice], *, size: int = _CHART_SIZE) -> bool:
    """Single pie with title and swatch legend."""
    if not slices:
        return False

    ensure_vega_chart_transparent_css()
    st.subheader(title)
    with st.container(horizontal_alignment="center"):
        st.altair_chart(
            _pie_chart(slices, size=size),
            width="content",
            height="content",
            key=f"pie_{''.join(c if c.isalnum() else '_' for c in title.lower()).strip('_')}",
        )
        render_series_legend(
            pie_legend_items(slices),
            marker="swatch",
            align="center",
            min_height_px=_LEGEND_MIN_HEIGHT,
        )
    return True


def render_pie_charts_row(
    charts: list[tuple[str, list[PieChartSlice]]],
    *,
    size: int = _CHART_SIZE,
) -> None:
    """Two pies side by side with a vertical divider column between them."""
    charts = [(title, slices) for title, slices in charts if slices]
    if not charts:
        return
    if len(charts) == 1:
        render_pie_chart(charts[0][0], charts[0][1], size=size)
        return

    divider_min_height = size + _LEGEND_MIN_HEIGHT + _TITLE_AND_GAPS
    with three_column_row(_ROW_WIDTHS, row_key=_ROW_KEY) as (col_left, col_divider, col_right):
        with col_left:
            render_pie_chart(charts[0][0], charts[0][1], size=size)
        with col_divider:
            render_center_divider_column(min_height_px=divider_min_height)
        with col_right:
            render_pie_chart(charts[1][0], charts[1][1], size=size)
