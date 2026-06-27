import streamlit as st

from chess_teacher.pipelines.runner import run_pipeline
from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.pipeline_utils.pipeline_helpers import aggregate_pipeline_run_results
from streamlit_utils.login import require_authenticated_user
from streamlit_utils.page_config import configure_page
from streamlit_utils.page_logging import log_page_view, log_user_action
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

_PIPELINE_RUN_ONCE_KEY = "pipeline_run_once"
_PIPELINE_RUNNING_KEY = "pipeline_running"
_PIPELINE_RESULT_KEY = "pipeline_result"
_PIPELINE_INTERRUPTED_KEY = "pipeline_interrupted"

st.session_state.setdefault(_PIPELINE_RUNNING_KEY, False)

should_run = st.session_state.pop(_PIPELINE_RUN_ONCE_KEY, False)
if st.session_state[_PIPELINE_RUNNING_KEY] and not should_run:
    st.session_state[_PIPELINE_RUNNING_KEY] = False
    st.session_state[_PIPELINE_INTERRUPTED_KEY] = True

if st.session_state.pop(_PIPELINE_INTERRUPTED_KEY, False):
    logger.warning(
        "Pipeline run interrupted by leaving page user_id=%s",
        user.user_id,
    )
    st.warning(
        "Previous pipeline run did not finish (you left this page). You can start a new run."
    )

accounts = user.get_linked_accounts(db_client)
pipeline_running = st.session_state[_PIPELINE_RUNNING_KEY]

if not accounts:
    st.info("There are no platform accounts linked.")

st.caption(
    "Run ingestion then preprocessing for every linked account, matching the scheduled worker job."
)

with st.form("pipeline_form"):
    submitted = st.form_submit_button(
        "Run pipeline",
        disabled=not accounts or pipeline_running,
    )

saved_result: ProgressSnapshot | None = st.session_state.get(_PIPELINE_RESULT_KEY)
if saved_result is not None and not pipeline_running:
    render_progress_snapshot(saved_result)

if submitted and not pipeline_running:
    st.session_state[_PIPELINE_RUN_ONCE_KEY] = True
    st.session_state.pop(_PIPELINE_RESULT_KEY, None)
    st.rerun()

if should_run:
    st.session_state[_PIPELINE_RUNNING_KEY] = True
    log_user_action(
        "Pipeline run started from Streamlit",
        user,
        linked_accounts=len(accounts),
    )

    with StreamlitProgressWindow() as progress:
        try:
            results = run_pipeline(user, db_client, progress_window=progress)
            aggregated = aggregate_pipeline_run_results(results)
            log_user_action(
                "Pipeline run finished from Streamlit",
                user,
                result=aggregated.result.value,
                run_count=len(aggregated.run_results),
            )
            if aggregated.latest_successful_run_id is not None:
                updated_user = user.update_latest_pipeline_run(
                    db_client,
                    aggregated.latest_successful_run_id,
                )
                set_current_user(updated_user)
        except Exception:
            logger.exception("Pipeline failed from Streamlit page user_id=%s", user.user_id)
        finally:
            st.session_state[_PIPELINE_RESULT_KEY] = progress.snapshot()
            st.session_state[_PIPELINE_RUNNING_KEY] = False
            st.rerun()
