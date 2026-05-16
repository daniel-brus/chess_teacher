"""Chess Teacher Streamlit Application."""

import streamlit as st

from streamlit_utils.login import LoginScreen
from streamlit_utils.session_state import force_logout, get_current_user

st.set_page_config(
    page_title="Chess Teacher",
    page_icon="♟️",
    layout="centered",
)

LoginScreen().display()
if "current_user" not in st.session_state.keys():
    st.stop()
user = get_current_user()

pages = [
    st.Page("pages/home.py", title="Home"),
    st.Page("pages/play.py", title="Play"),
    st.Page("pages/settings.py", title="Settings"),
]

pg = st.navigation(pages, position="sidebar", expanded=False)

with st.sidebar:
    if user.picture:
        st.image(user.picture)
    else:
        st.markdown(":chess_pawn:")

    if st.button("Logout"):
        force_logout()  # clear + rerun → LoginScreen pakt het op

pg.run()
