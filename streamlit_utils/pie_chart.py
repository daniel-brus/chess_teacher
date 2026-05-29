"""Fixed-size Altair pie charts with HTML legend below (Streamlit)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import altair as alt
import streamlit as st

from chess_teacher.platform.account import Account, AccountPlatform
from streamlit_utils.chart_legend import SeriesLegendItem, render_series_legend, series_colors
from streamlit_utils.layout import column_with_divider

PIE_SIZE = 168


@dataclass(frozen=True)
class PieChartSlice:
    label: str
    value: float
    color: str
    logo_path: Path | None = None
    platform: AccountPlatform | None = None


def _legend_items(slices: list[PieChartSlice]) -> list[SeriesLegendItem]:
    return [
        SeriesLegendItem(
            series_id=slice_.label,
            color=slice_.color,
            label=slice_.label,
            platform=slice_.platform,
            logo_path=slice_.logo_path,
        )
        for slice_ in sorted(slices, key=lambda slice_: slice_.label)
    ]


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


def account_pie_slices(
    rows: list[dict[str, object]],
    accounts_by_id: dict[str, Account],
) -> list[PieChartSlice]:
    label_to_account = {account.format_label(): account for account in accounts_by_id.values()}
    counts = {str(row["account"]): float(row["games"]) for row in rows}  # type: ignore[arg-type]
    labels = sorted(counts)
    colors = series_colors(labels)
    return [
        PieChartSlice(
            label=label,
            value=counts[label],
            color=colors[label],
            platform=(account.platform if (account := label_to_account.get(label)) else None),
            logo_path=(account.platform.logo_path() if account else None),
        )
        for label in labels
    ]


def _pie_chart_key(title: str) -> str:
    slug = "".join(char if char.isalnum() else "_" for char in title.lower())
    return f"pie_{slug.strip('_')}"


def render_pie_chart(
    title: str,
    slices: list[PieChartSlice],
    *,
    column: st.delta_generator.DeltaGenerator | None = None,
    size: int = PIE_SIZE,
) -> bool:
    """Draw pie + HTML swatch legend. Chart area is fixed ``size`` x ``size`` px."""
    if not slices:
        return False

    domain = [slice_.label for slice_ in slices]
    chart = (
        alt
        .Chart(
            alt.Data(
                values=[{"category": slice_.label, "value": slice_.value} for slice_ in slices]
            )
        )
        .mark_arc()
        .encode(
            theta=alt.Theta("value:Q", stack=True),
            color=alt.Color(
                "category:N",
                scale=alt.Scale(
                    domain=domain,
                    range=[slice_.color for slice_ in slices],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("category:N", title="Category"),
                alt.Tooltip("value:Q", title="Games"),
            ],
        )
        .properties(width=size, height=size)
        .configure_view(strokeWidth=0)
    )

    def _draw() -> None:
        with st.container(horizontal_alignment="center"):
            st.subheader(title)
            st.altair_chart(chart, width=size, key=_pie_chart_key(title))
            render_series_legend(_legend_items(slices), marker="swatch", align="center")

    if column is not None:
        with column:
            _draw()
    else:
        _draw()
    return True


def render_pie_charts_row(
    charts: list[tuple[str, list[PieChartSlice]]],
    *,
    size: int = PIE_SIZE,
) -> None:
    """Render one or two pies; two pies get a vertical divider between them."""
    charts = [(title, slices) for title, slices in charts if slices]
    if not charts:
        return
    if len(charts) == 1:
        render_pie_chart(charts[0][0], charts[0][1], size=size)
        return

    col_left, col_right = st.columns(2, gap="small")
    with col_left:
        with column_with_divider("pie_left"):
            render_pie_chart(charts[0][0], charts[0][1], size=size)
    render_pie_chart(charts[1][0], charts[1][1], column=col_right, size=size)
