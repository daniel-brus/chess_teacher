"""HTML chart legends (icons + line colors) for Streamlit + Altair."""

from __future__ import annotations

import base64
import html
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl
import streamlit as st

from chess_teacher.platform.account import Account, AccountPlatform

# Vega category10 — keep in sync with Altair ``scheme="category10"`` on rating chart.
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


@dataclass(frozen=True)
class SeriesLegendItem:
    series_id: str
    color: str
    label: str
    platform: AccountPlatform | None
    logo_path: Path | None


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
    items: list[SeriesLegendItem] = []
    for row in meta.iter_rows(named=True):
        account = accounts_by_id.get(row["account_id"])
        platform = account.platform if account else None
        logo_path = platform.logo_path() if platform else None
        items.append(
            SeriesLegendItem(
                series_id=row["series_id"],
                color=colors[row["series_id"]],
                label=row["series_label"],
                platform=platform,
                logo_path=logo_path,
            )
        )
    return items


LegendMarker = Literal["line", "swatch"]


def _svg_data_uri(path: Path) -> str | None:
    if not path.is_file():
        return None
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def render_series_legend(
    items: list[SeriesLegendItem],
    *,
    marker: LegendMarker = "line",
    align: Literal["left", "center"] = "left",
) -> None:
    """Legend below chart: color marker + optional platform SVG + label."""
    if not items:
        return

    parts: list[str] = []
    icon_size = 14 if marker == "swatch" else 16
    for item in items:
        icon_html = ""
        if item.logo_path is not None:
            data_uri = _svg_data_uri(item.logo_path)
            if data_uri:
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
        parts.append(
            "<span style="
            '"display:inline-flex;align-items:center;margin:2px 10px 2px 0;">'
            f"{color_marker}"
            f"{icon_html}"
            f"<span>{html.escape(item.label)}</span>"
            "</span>"
        )

    font_size = "0.82rem" if marker == "swatch" else "0.9rem"
    justify = "center" if align == "center" else "flex-start"
    st.html(
        f'<div style="display:flex;flex-wrap:wrap;align-items:center;'
        f'justify-content:{justify};font-size:{font_size};width:100%;">' + "".join(parts) + "</div>"
    )
