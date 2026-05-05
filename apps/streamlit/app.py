"""Chess Teacher Streamlit Application."""

import streamlit as st
from chess_teacher.utils.logging_utils import get_logger

logger = get_logger()

st.set_page_config(
    page_title="Chess Teacher",
    page_icon="♟️",
    layout="centered",
)

logger.info("Chess Teacher Streamlit app started")

st.title("♟️ Chess Teacher")
st.write("Welkom!")
