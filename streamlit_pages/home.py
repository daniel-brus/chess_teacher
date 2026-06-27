import streamlit as st

from streamlit_utils.login import require_authenticated_user
from streamlit_utils.page_config import configure_page
from streamlit_utils.page_logging import log_page_view
from streamlit_utils.platform_ui import render_app_logo

configure_page("Home")
user = require_authenticated_user()
log_page_view("Home", user)

st.title(f"Welcome to the Chess Teacher app, {user.name}!")
render_app_logo()

st.markdown("todo")
