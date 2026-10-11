import json
import time

import streamlit as st

from chess_teacher.platform.user import User
from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.env_utils import get_env_variable
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.pipeline_utils.json_lines_progress import apply_progress_event
from orchestration.pipeline_jobs import (
    create_pipeline_job,
    get_pipeline_job,
    list_user_pipeline_jobs,
    pipeline_job_state,
    read_pipeline_job_logs,
)
from streamlit_utils.login import require_authenticated_user
from streamlit_utils.page_config import configure_page
from streamlit_utils.page_logging import log_page_view, log_user_action
from streamlit_utils.progress_window import (
    ProgressSnapshot,
    StreamlitProgressWindow,
    render_progress_snapshot,
)
from streamlit_utils.session_state import set_current_user


def _apply_progress_log_line(progress: StreamlitProgressWindow, line: str) -> None:
    logger = get_logger()
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return
    if not isinstance(event, dict) or "op" not in event:
        return
    try:
        apply_progress_event(progress, event)
    except ValueError as exc:
        logger.warning("Ignoring unrecognized pipeline progress event: %s", exc)


configure_page("Pipeline")

db_client = get_db_client()
logger = get_logger()
user = require_authenticated_user()
log_page_view("Pipeline", user)

st.title("Run the pipeline")

_PIPELINE_RESULT_KEY = "pipeline_result"

accounts = user.get_linked_accounts(db_client)
namespace = get_env_variable("K8S_NAMESPACE") if accounts else ""
user_jobs = list_user_pipeline_jobs(namespace=namespace, user_id=user.user_id) if accounts else []
active_job = next(
    (job for job in user_jobs if pipeline_job_state(job)[0] in {"pending", "running"}),
    None,
)

st.caption(
    "Run ingestion, preprocessing, and training in a separate Kubernetes worker. "
    "Refreshing this page does not stop a run that is already in progress."
)

if active_job is not None:
    st.info("This run keeps going if you refresh or leave the page.")

if not accounts:
    st.info("There are no platform accounts linked.")

with st.form("pipeline_form"):
    submitted = st.form_submit_button(
        "Run pipeline",
        disabled=not accounts or active_job is not None,
    )

saved_result: ProgressSnapshot | None = st.session_state.get(_PIPELINE_RESULT_KEY)
if submitted and active_job is None:
    log_user_action(
        "Pipeline run started from Streamlit",
        user,
        linked_accounts=len(accounts),
    )
    job_name = create_pipeline_job(namespace=namespace, user_id=user.user_id)
    logger.info("Kubernetes pipeline job started user_id=%s job=%s", user.user_id, job_name)
    st.rerun()

if active_job is not None:
    job_name = active_job.metadata.name
    st.caption(f"Worker job: {job_name}")
    with StreamlitProgressWindow() as progress:
        progress.set_pacing(False)
        previous_logs = ""
        progress.next("Waiting for the worker to start...")
        while True:
            logs = read_pipeline_job_logs(namespace=namespace, job_name=job_name)
            if logs.startswith(previous_logs):
                new_logs = logs[len(previous_logs) :]
            else:
                new_logs = logs
            previous_logs = logs
            for line in new_logs.splitlines():
                _apply_progress_log_line(progress, line)

            job = get_pipeline_job(namespace=namespace, job_name=job_name)
            state, message = pipeline_job_state(job)
            if state == "succeeded":
                progress.success("Pipeline completed.")
                break
            if state == "failed":
                progress.error(message or "Pipeline worker failed.")
                break

            progress.checkpoint()
            time.sleep(1)

    st.session_state[_PIPELINE_RESULT_KEY] = progress.snapshot()
    if state == "succeeded":
        log_user_action(
            "Pipeline run finished from Streamlit",
            user,
            linked_accounts=len(accounts),
        )
        set_current_user(User.fetch_from_db(db_client, id=user.user_id))
    logger.info(
        "Kubernetes pipeline watcher finished user_id=%s job=%s state=%s",
        user.user_id,
        job_name,
        state,
    )
    st.rerun()
elif user_jobs:
    latest_job = user_jobs[0]
    latest_state, message = pipeline_job_state(latest_job)
    with StreamlitProgressWindow() as progress:
        progress.set_pacing(False)
        for line in read_pipeline_job_logs(
            namespace=namespace,
            job_name=latest_job.metadata.name,
        ).splitlines():
            _apply_progress_log_line(progress, line)
        if latest_state == "succeeded":
            progress.success("Pipeline completed.")
        elif latest_state == "failed":
            progress.error(message or "Pipeline worker failed.")
        else:
            progress.next("Pipeline worker is waiting to start.")
    st.session_state[_PIPELINE_RESULT_KEY] = progress.snapshot()
elif saved_result is not None:
    render_progress_snapshot(saved_result)
