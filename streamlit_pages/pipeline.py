import streamlit as st

from chess_teacher.platform.user import User
from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.logging import get_logger
from streamlit_utils.login import require_authenticated_user
from streamlit_utils.page_config import configure_page
from streamlit_utils.page_logging import log_page_view, log_user_action
from streamlit_utils.pipeline_subprocess import (
    clear_pipeline_run,
    find_pipeline_run,
    follow_pipeline,
    pipeline_is_running,
    start_detached_pipeline,
)
from streamlit_utils.progress_window import (
    ProgressSnapshot,
    StreamlitProgressWindow,
    render_progress_snapshot,
)
from streamlit_utils.session_state import set_current_user

configure_page("Pipeline")

db_client = get_db_client()
logger = get_logger()
user = require_authenticated_user()
log_page_view("Pipeline", user)

st.title("Run the pipeline")

_PIPELINE_RESULT_KEY = "pipeline_result"

accounts = user.get_linked_accounts(db_client)
active_run = find_pipeline_run(user.user_id)
running = active_run is not None and pipeline_is_running(active_run)

st.caption(
    "Run ingestion and preprocessing for every linked account, then train this user's model, "
    "matching the scheduled worker job. Refreshing this page does not stop a run that is already going."
)

if running:
    st.info("This run keeps going if you refresh or leave the page.")

if not accounts:
    st.info("There are no platform accounts linked.")

with st.form("pipeline_form"):
    submitted = st.form_submit_button(
        "Run pipeline",
        disabled=not accounts or active_run is not None,
    )

saved_result: ProgressSnapshot | None = st.session_state.get(_PIPELINE_RESULT_KEY)
if saved_result is not None and active_run is None:
    render_progress_snapshot(saved_result)

if submitted and active_run is None:
    log_user_action(
        "Pipeline run started from Streamlit",
        user,
        linked_accounts=len(accounts),
    )
    start_detached_pipeline(user.user_id)
    logger.info("Detached pipeline supervisor started user_id=%s", user.user_id)
    st.rerun()

if active_run is not None:
    with StreamlitProgressWindow() as progress:
        exit_code = follow_pipeline(active_run, progress)
    clear_pipeline_run(user.user_id)
    if exit_code == 0:
        log_user_action(
            "Pipeline run finished from Streamlit",
            user,
            linked_accounts=len(accounts),
        )
        set_current_user(User.fetch_from_db(db_client, id=user.user_id))
    logger.info(
        "Detached pipeline watcher finished user_id=%s exit_code=%s",
        user.user_id,
        exit_code,
    )
    st.session_state[_PIPELINE_RESULT_KEY] = progress.snapshot()
    st.rerun()
