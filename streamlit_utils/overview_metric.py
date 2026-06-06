"""Compact overview stat cards with dynamic value sizing (replaces ``st.metric``)."""

from __future__ import annotations

import html
from collections.abc import Sequence

import streamlit as st

from streamlit_utils.layout import ingest_css

_CSS_SESSION_KEY = "_ct_overview_metrics_css"


def overview_value_font_rem(text: str) -> float:
    """Smaller rem when the primary value is longer so it fits the column."""
    length = len(text)
    if length <= 5:
        return 1.35
    if length <= 8:
        return 1.2
    if length <= 12:
        return 1.05
    if length <= 18:
        return 0.95
    if length <= 28:
        return 0.85
    return 0.75


def ingest_overview_metrics_css() -> None:
    if st.session_state.get(_CSS_SESSION_KEY):
        return
    st.session_state[_CSS_SESSION_KEY] = True
    ingest_css(
        """
<style>
.ct-overview-metric {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    min-height: 5.25rem;
    padding: 0.15rem 0.35rem 0.35rem 0;
    box-sizing: border-box;
}
.ct-overview-label {
    font-size: 0.8125rem;
    line-height: 1.25;
    color: color-mix(in srgb, var(--text-color) 68%, transparent);
    word-wrap: break-word;
}
.ct-overview-help {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 0.95rem;
    height: 0.95rem;
    margin-left: 0.2rem;
    border-radius: 50%;
    border: 1px solid color-mix(in srgb, var(--text-color) 35%, transparent);
    font-size: 0.625rem;
    font-weight: 600;
    vertical-align: middle;
    cursor: help;
}
.ct-overview-value {
    font-weight: 600;
    line-height: 1.2;
    color: var(--text-color);
    overflow-wrap: anywhere;
    word-break: break-word;
    white-space: normal;
    max-width: 100%;
}
.ct-overview-detail {
    font-size: 0.75rem;
    line-height: 1.3;
    color: color-mix(in srgb, var(--text-color) 62%, transparent);
    overflow-wrap: anywhere;
    word-break: break-word;
}
.ct-overview-section-title {
    font-size: 1rem;
    font-weight: 600;
    line-height: 1.25;
    margin: 0 0 0.35rem 0;
    padding: 0.15rem 0.35rem 0;
    color: var(--text-color);
}
</style>
"""
    )


def render_overview_section_title(title: str) -> None:
    """Column heading for grouped overview stats (e.g. White / Black)."""
    st.html(f'<div class="ct-overview-section-title">{html.escape(title)}</div>')


def render_overview_metric(
    label: str,
    value: str,
    *,
    details: Sequence[str] = (),
    help: str | None = None,
) -> None:
    """Render one overview stat in the current column.

    Call :func:`ingest_overview_metrics_css` once before the metric grid (not inside a column).
    """
    safe_label = html.escape(label)
    safe_value = html.escape(value)
    font_rem = overview_value_font_rem(value)

    help_html = ""
    if help:
        safe_help = html.escape(help, quote=True)
        help_html = (
            f'<span class="ct-overview-help" title="{safe_help}" aria-label="{safe_help}">?</span>'
        )

    detail_lines = "".join(
        f'<div class="ct-overview-detail">{html.escape(line)}</div>' for line in details
    )

    st.html(
        f"""
<div class="ct-overview-metric">
  <div class="ct-overview-label">{safe_label}{help_html}</div>
  <div class="ct-overview-value" style="font-size:{font_rem}rem">{safe_value}</div>
  {detail_lines}
</div>
"""
    )
