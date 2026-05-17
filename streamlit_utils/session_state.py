import streamlit as st

from chess_teacher.platform.user import User
from chess_teacher.utils.exception_utils import AuthError
from chess_teacher.utils.logging_utils import get_logger

logger = get_logger()


def force_logout() -> None:
    st.session_state.clear()
    st.logout()


def st_user_is_logged_in() -> bool:
    """Check if the user is logged in."""
    try:
        return st.user.is_logged_in
    except Exception:
        logger.warning(AuthError("Failed to check login status"), exc_info=True)
        return False


def get_current_user() -> User:
    if "current_user" not in st.session_state:
        st.stop()
    try:
        return st.session_state["current_user"]
    except Exception as e:
        logger.log_and_raise(e)


def set_current_user(user: User) -> None:
    try:
        st.session_state["current_user"] = user
    except Exception as e:
        logger.log_and_raise(e)
