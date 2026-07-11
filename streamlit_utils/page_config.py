"""Shared ``st.set_page_config`` for the app shell and each page."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import streamlit as st

from streamlit_utils.theme import reset_theme_css_state

APP_NAME = "Chess Teacher"
_DEFAULT_FAVICON = "♟️"
_DEFAULT_LAYOUT: Literal["centered", "wide"] = "centered"
_DEFAULT_SIDEBAR: Literal["auto", "expanded", "collapsed"] = "collapsed"


def _resolve_page_icon(page_icon: Path | str | None) -> Path | str:
    if page_icon is not None:
        return page_icon
    # Streamlit opens ``page_icon`` bytes with PIL; our wordmark is SVG (see render_app_logo).
    return _DEFAULT_FAVICON


def configure_page(
    page_title: str | None = None,
    *,
    page_icon: Path | str | None = None,
    layout: Literal["centered", "wide"] = _DEFAULT_LAYOUT,
    initial_sidebar_state: Literal["auto", "expanded", "collapsed"] = _DEFAULT_SIDEBAR,
    menu_items: dict[str, str] | None = None,
) -> None:
    """Call once at the top of ``streamlit_app.py`` and every ``streamlit_pages/*.py`` script.

    ``page_title`` becomes ``"{page_title} | Chess Teacher"`` in the browser tab.
    Omit ``page_title`` on the entry script for a tab title of ``"Chess Teacher"`` only.
    """
    if page_title is None:
        reset_theme_css_state()

    if page_title:
        browser_title = f"{page_title} | {APP_NAME}"
    else:
        browser_title = APP_NAME

    st.set_page_config(
        page_title=browser_title,
        page_icon=_resolve_page_icon(page_icon),
        layout=layout,
        initial_sidebar_state=initial_sidebar_state,
        menu_items=menu_items
        or {
            "About": (f"{APP_NAME} — ingest and analyze your games from Chess.com and Lichess."),
        },
    )
