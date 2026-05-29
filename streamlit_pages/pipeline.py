import streamlit as st

from chess_teacher.ingestion.main import run_ingestion_pipeline
from chess_teacher.platform.users_accounts import get_accounts_for_user
from chess_teacher.utils.db_client import get_db_client
from chess_teacher.utils.logging_utils import get_logger
from streamlit_utils.login import require_authenticated_user
from streamlit_utils.platform_ui import pick_one_account
from streamlit_utils.progress_window import (
    ProgressSnapshot,
    StreamlitProgressWindow,
    render_progress_snapshot,
)

db_client = get_db_client()
logger = get_logger()

user = require_authenticated_user()

st.title("Run the pipeline")

_PIPELINE_RUN_ONCE_KEY = "pipeline_run_once"
_PIPELINE_RUNNING_KEY = "pipeline_running"
_PIPELINE_ACCOUNT_KEY = "pipeline_account"
_PIPELINE_RESULT_KEY = "pipeline_result"
_PIPELINE_INTERRUPTED_KEY = "pipeline_interrupted"

st.session_state.setdefault(_PIPELINE_RUNNING_KEY, False)

should_run = st.session_state.pop(_PIPELINE_RUN_ONCE_KEY, False)
if st.session_state[_PIPELINE_RUNNING_KEY] and not should_run:
    st.session_state[_PIPELINE_RUNNING_KEY] = False
    st.session_state[_PIPELINE_INTERRUPTED_KEY] = True

if st.session_state.pop(_PIPELINE_INTERRUPTED_KEY, False):
    st.warning(
        "Previous pipeline run did not finish (you left this page). You can start a new run."
    )

accounts = get_accounts_for_user(user, db_client)
pipeline_running = st.session_state[_PIPELINE_RUNNING_KEY]
running_account = st.session_state.get(_PIPELINE_ACCOUNT_KEY)

selected_account = pick_one_account(
    accounts,
    label="Account",
    key_prefix="pipeline_account_select",
    default=running_account,
    disabled=not accounts,
    allow_change=not pipeline_running,
)

with st.form("pipeline_form"):
    submitted = st.form_submit_button(
        "Run pipeline",
        disabled=not accounts or not selected_account or pipeline_running,
    )

if not accounts:
    st.info("There are no platform accounts linked.")

saved_result: ProgressSnapshot | None = st.session_state.get(_PIPELINE_RESULT_KEY)
if saved_result is not None and not pipeline_running:
    render_progress_snapshot(saved_result)

if submitted and selected_account and not pipeline_running:
    st.session_state[_PIPELINE_ACCOUNT_KEY] = selected_account
    st.session_state[_PIPELINE_RUN_ONCE_KEY] = True
    st.session_state.pop(_PIPELINE_RESULT_KEY, None)
    st.rerun()

if should_run:
    st.session_state[_PIPELINE_RUNNING_KEY] = True
    with StreamlitProgressWindow() as progress:
        try:
            run_ingestion_pipeline(
                user.user_id,
                st.session_state[_PIPELINE_ACCOUNT_KEY],
                progress_window=progress,
            )
        except Exception:
            logger.error("Pipeline failed from Streamlit page.")
        finally:
            st.session_state[_PIPELINE_RESULT_KEY] = progress.snapshot()
            st.session_state[_PIPELINE_RUNNING_KEY] = False
            st.rerun()
