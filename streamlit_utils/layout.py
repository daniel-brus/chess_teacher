"""Shared Streamlit layout helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import streamlit as st

from streamlit_utils.theme import divider_border, divider_rgba, divider_width_px

# Each ``st.markdown("<style>…")`` becomes a vertical-block sibling; Streamlit adds flex
# gap between siblings. Hide those slots so only real widgets affect layout spacing.
_COLLAPSE_STYLE_ONLY_ELEMENTS_CSS = """
[data-testid="stElementContainer"]:has([data-testid="stMarkdownContainer"] style),
[data-testid="stElementContainer"]:has([data-testid="stHtml"] style) {
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}
"""


def ingest_css(css: str) -> None:
    """Inject CSS. Style-only elements are collapsed via :func:`_collapse_style_only_elements_css`."""
    body = css.strip()
    if "<style" not in body:
        body = f"<style>\n{body}\n</style>"
    st.markdown(body, unsafe_allow_html=True)


def shell_css() -> str:
    """Layout CSS shared across pages (not palette-specific)."""
    return (
        _collapse_style_only_elements_css()
        + main_block_top_padding_css()
        + sidebar_bottom_logout_css()
    )


def _collapse_style_only_elements_css() -> str:
    return _COLLAPSE_STYLE_ONLY_ELEMENTS_CSS


def main_block_top_padding_css(*, padding_top: str = "2rem") -> str:
    """CSS fragment: tight top padding on the main content block."""
    return f"""
[data-testid="stMainBlockContainer"],
section.main div.block-container {{
    padding-top: {padding_top} !important;
}}
[data-testid="stMainBlockContainer"] h1:first-of-type {{
    margin-top: 0;
}}
"""


def sidebar_bottom_logout_css() -> str:
    """CSS fragment: pin ``sidebar_logout`` to the bottom of the sidebar."""
    return """
section[data-testid="stSidebar"] > div:first-child {
    height: 100vh;
}

section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    position: relative;
    height: 100%;
}

section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding-bottom: 5rem;
    box-sizing: border-box;
}

div[class*="st-key-sidebar_logout"] {
    position: absolute !important;
    bottom: 1rem !important;
    left: 0 !important;
    right: 0 !important;
    z-index: 2;
    margin: 0 !important;
    padding: 0 !important;
}

div[class*="st-key-sidebar_logout"] button {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

div[class*="st-key-sidebar_logout"] button [data-testid="stMarkdownContainer"],
div[class*="st-key-sidebar_logout"] button [data-testid="stMarkdownContainer"] p {
    margin: 0 !important;
    line-height: 1 !important;
}
"""


def ingest_row_stretch_css(row_key: str) -> None:
    """Stretch all columns in a ``st.columns`` row to the same height."""
    ingest_css(
        f"""
<style>
div[class*="st-key-{row_key}"] [data-testid="stHorizontalBlock"] {{
    align-items: stretch !important;
}}
</style>
"""
    )


def render_center_divider_column(*, min_height_px: int, row_key: str | None = None) -> None:
    """Content for the narrow middle column between two panels (vertical rule)."""
    if row_key is not None:
        ingest_row_stretch_css(row_key)
    ingest_css(
        """
<style>
div[class*="st-key-center_divider"] {
    display: flex;
    justify-content: center;
    height: 100%;
    min-height: 100%;
}
</style>
"""
    )
    with st.container(key="center_divider"):
        st.html(
            f'<div style="width:{divider_width_px()}px;min-height:{min_height_px}px;'
            f'height:100%;background:{divider_rgba()};" aria-hidden="true"></div>'
        )


@contextmanager
def three_column_row(
    column_widths: tuple[int, ...],
    *,
    row_key: str,
    gap: str = "small",
) -> Iterator[tuple[st.delta_generator.DeltaGenerator, ...]]:
    """``st.columns`` row with equal-height columns (for pie | divider | pie)."""
    ingest_row_stretch_css(row_key)
    with st.container(key=row_key):
        yield st.columns(column_widths, gap=gap)


def ingest_column_divider_css(key: str) -> None:
    """Border-right on a keyed container. Call before columns, not inside one."""
    ingest_css(
        f"""
<style>
div[class*="st-key-{key}"] {{
    border-right: {divider_border()};
}}
</style>
"""
    )


@contextmanager
def column_with_divider(key: str) -> Iterator[None]:
    """Keyed container with border (register CSS via :func:`ingest_column_divider_css` first)."""
    with st.container(key=key):
        yield
