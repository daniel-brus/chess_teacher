"""Chess Teacher Streamlit Application."""

import streamlit as st

from streamlit_utils.login import require_authenticated_user
from streamlit_utils.session_state import force_logout

st.set_page_config(
    page_title="Chess Teacher",
    page_icon="♟️",
    layout="centered",
)

user = require_authenticated_user()

pages = [
    st.Page("views/home.py", title="Home"),
    st.Page("views/pipeline.py", title="Pipeline"),
    st.Page("views/play.py", title="Play"),
    st.Page("views/settings.py", title="Settings"),
]

pg = st.navigation(pages, position="sidebar", expanded=False)

with st.sidebar:
    if user.picture:
        st.image(user.picture)
    else:
        st.markdown(":chess_pawn:")

    if st.button("Logout"):
        force_logout()

pg.run()
