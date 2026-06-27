import streamlit as st

from streamlit_utils.login import require_authenticated_user
from streamlit_utils.page_config import configure_page
from streamlit_utils.page_logging import log_page_view, log_user_action

configure_page("Play")
user = require_authenticated_user()
log_page_view("Play", user)

st.title("Play a game of chess")

st.text("Not yet implemented")

if st.button("Win a game!"):
    log_user_action("Play page demo button clicked", user)
    st.balloons()
