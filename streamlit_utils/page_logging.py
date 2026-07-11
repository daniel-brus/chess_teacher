"""Logging helpers for Streamlit navigation and user actions."""

from __future__ import annotations

import streamlit as st

from chess_teacher.platform.user import User
from chess_teacher.utils.logging import get_logger

logger = get_logger()


def log_page_view(page: str, user: User) -> None:
    """Log when the user opens ``page`` (once per visit, not on every widget rerun)."""
    active = st.session_state.get("_streamlit_active_page")
    if active == page:
        return
    st.session_state["_streamlit_active_page"] = page
    logger.info("Streamlit page %s user_id=%s", page, user.user_id)


def log_user_action(message: str, user: User, **context: object) -> None:
    """Log a deliberate user action from a Streamlit page."""
    details = " ".join(f"{key}={value!r}" for key, value in context.items())
    suffix = f" {details}" if details else ""
    logger.info("%s user_id=%s%s", message, user.user_id, suffix)
