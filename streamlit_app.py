"""Chess Teacher Streamlit Application."""

import streamlit as st

from streamlit_utils.login import require_authenticated_user
from streamlit_utils.page_config import configure_page
from streamlit_utils.profile_ui import render_sidebar_profile
from streamlit_utils.session_state import force_logout

configure_page()

user = require_authenticated_user()

pages = [
    st.Page("streamlit_pages/home.py", title="Home"),
    st.Page("streamlit_pages/pipeline.py", title="Pipeline"),
    st.Page("streamlit_pages/play.py", title="Play"),
    st.Page("streamlit_pages/statistics.py", title="Statistics"),
    st.Page("streamlit_pages/settings.py", title="Settings"),
]

pg = st.navigation(pages, position="hidden")

with st.sidebar:
    render_sidebar_profile(user)
    for page in pages:
        st.page_link(page, width="stretch")
    with st.container(key="sidebar_logout"):
        if st.button("Logout", width="stretch"):
            force_logout()

pg.run()
