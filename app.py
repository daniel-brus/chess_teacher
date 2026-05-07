"""Chess Teacher Streamlit Application."""

import streamlit as st

from chess_teacher.utils.logging_utils import get_logger
from streamlit_utils.auth import login_screen

logger = get_logger()

st.set_page_config(
    page_title="Chess Teacher",
    page_icon="♟️",
    layout="centered",
)

logger.info("Chess Teacher Streamlit app started")

st.title("♟️ Chess Teacher")

if not st.user.get("is_logged_in", False):
    login_screen()
else:
    st.write(f"Welcome, {st.user.get("name", "User")}!")
    st.button("Log out", on_click=st.logout)
