import streamlit as st

from streamlit_utils.login import require_authenticated_user
from streamlit_utils.page_config import configure_page

configure_page("Play")
require_authenticated_user()

st.title("Play a game of chess")

st.text("Not yet implemented")

if st.button("Win a game!"):
    st.balloons()
