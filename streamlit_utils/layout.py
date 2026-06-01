"""Shared Streamlit layout helpers."""

from __future__ import annotations

import base64
from collections.abc import Iterator
from contextlib import contextmanager

import streamlit as st


@contextmanager
def column_with_divider(key: str) -> Iterator[None]:
    """Wrap column content; Streamlit ``key`` gets a full-height border-right."""
    _divider_css = """
<style>
div[class*="st-key-col_divider_"] {
    border-right: 1px solid rgba(49, 51, 63, 0.25);
}
</style>
"""
    st.markdown(_divider_css, unsafe_allow_html=True)
    with st.container(key=f"col_divider_{key}"):
        yield


def set_bg(color1: str, color2: str, square_size: int = 40):
    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="{square_size * 2}" height="{square_size * 2}">
      <rect width="{square_size}" height="{square_size}" x="0" y="0" fill="{color1}"/>
      <rect width="{square_size}" height="{square_size}" x="{square_size}" y="0" fill="{color2}"/>
      <rect width="{square_size}" height="{square_size}" x="0" y="{square_size}" fill="{color2}"/>
      <rect width="{square_size}" height="{square_size}" x="{square_size}" y="{square_size}" fill="{color1}"/>
    </svg>
    """
    encoded = base64.b64encode(svg.encode()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/svg+xml;base64,{encoded}");
            background-repeat: repeat;
            background-size: {square_size * 2}px {square_size * 2}px;
        }}
        </style>
    """,
        unsafe_allow_html=True,
    )
