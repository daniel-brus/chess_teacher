"""Chess Teacher Streamlit Application."""

import streamlit as st

from chess_teacher.platform.user import User
from streamlit_utils.admin_auth import is_admin
from streamlit_utils.legal import render_legal_links
from streamlit_utils.login import require_authenticated_user
from streamlit_utils.page_config import configure_page
from streamlit_utils.page_logging import log_user_action
from streamlit_utils.profile_ui import render_sidebar_profile
from streamlit_utils.session_state import force_logout, st_user_is_logged_in

configure_page()

privacy_page = st.Page(
    "streamlit_pages/privacy.py",
    title="Privacy policy",
    url_path="privacy",
)
terms_page = st.Page(
    "streamlit_pages/terms.py",
    title="Terms of use",
    url_path="terms",
)
legal_pages = [privacy_page, terms_page]

app_pages = [
    st.Page("streamlit_pages/home.py", title="Home", default=True),
    st.Page("streamlit_pages/pipeline.py", title="Pipeline"),
    st.Page("streamlit_pages/play.py", title="Play"),
    st.Page("streamlit_pages/statistics.py", title="Statistics"),
    st.Page("streamlit_pages/settings.py", title="Settings"),
]
admin_page = st.Page("streamlit_pages/admin.py", title="Admin")

pg = st.navigation([*app_pages, admin_page, *legal_pages], position="hidden")
is_legal = pg is privacy_page or pg is terms_page


def _render_authenticated_sidebar(user: User) -> None:
    with st.sidebar:
        render_sidebar_profile(user)
        for page in app_pages:
            st.page_link(page, width="stretch")
        if is_admin(user):
            st.page_link(admin_page, width="stretch")
        st.divider()
        render_legal_links()
        with st.container(key="sidebar_logout"):
            if st.button("Logout", width="stretch"):
                log_user_action("User logged out from Streamlit", user)
                force_logout()


if is_legal:
    if st_user_is_logged_in():
        _render_authenticated_sidebar(require_authenticated_user())
    pg.run()
else:
    user = require_authenticated_user()
    _render_authenticated_sidebar(user)
    pg.run()
