"""Shared Streamlit layout helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import streamlit as st

_DIVIDER_CSS = """
<style>
div[class*="st-key-col_divider_"] {
    border-right: 1px solid rgba(49, 51, 63, 0.25);
}
</style>
"""


def _inject_divider_css() -> None:
    st.markdown(_DIVIDER_CSS, unsafe_allow_html=True)


@contextmanager
def column_with_divider(key: str) -> Iterator[None]:
    """Wrap column content; Streamlit ``key`` gets a full-height border-right."""
    _inject_divider_css()
    with st.container(key=f"col_divider_{key}"):
        yield
