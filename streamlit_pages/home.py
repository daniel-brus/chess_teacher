import streamlit as st

from chess_teacher.utils.db.client import get_db_client
from streamlit_utils.layout import app_page_link
from streamlit_utils.login import require_authenticated_user
from streamlit_utils.page_config import APP_DESCRIPTION, configure_page
from streamlit_utils.page_logging import log_page_view
from streamlit_utils.platform_ui import render_app_logo

configure_page("Home")
user = require_authenticated_user()
log_page_view("Home", user)
db_client = get_db_client()

display_name = user.name or "chess player"
st.title(f"Welcome to Chess Teacher, {display_name}!")
render_app_logo()
st.markdown(APP_DESCRIPTION)

accounts = user.get_linked_accounts(db_client)
latest_run = user.get_latest_pipeline_run(db_client)

if not accounts:
    st.info("Start by linking a Chess.com or Lichess account, then import your games.")
    app_page_link(
        "streamlit_pages/settings.py",
        "Link a platform account",
    )
    app_page_link(
        "streamlit_pages/play.py",
        "Or play a game now",
    )
elif latest_run is None:
    st.info("Accounts are linked. Run the pipeline to import your public games.")
    app_page_link(
        "streamlit_pages/pipeline.py",
        "Run the pipeline",
    )
    app_page_link(
        "streamlit_pages/play.py",
        "Play a game",
    )
else:
    result_label = latest_run.result.value.replace("_", " ").title()
    finished = latest_run.finished_at.strftime("%d %b %Y, %H:%M UTC")
    st.success(f"Last pipeline run: {result_label} · {finished}")
    play_col, stats_col = st.columns(2)
    with play_col:
        app_page_link("streamlit_pages/play.py", "Play a game")
    with stats_col:
        app_page_link(
            "streamlit_pages/statistics.py",
            "View statistics",
        )
    app_page_link(
        "streamlit_pages/pipeline.py",
        "Run the pipeline again",
    )
