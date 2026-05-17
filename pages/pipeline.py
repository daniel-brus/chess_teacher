import html
import logging
from datetime import datetime

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from chess_teacher.platform.account import Account
from chess_teacher.platform.users_accounts import get_accounts_for_user
from chess_teacher.utils.db_client import get_db_client
from chess_teacher.utils.logging_utils import get_logger
from scripts.pipeline import run_pipeline
from streamlit_utils.session_state import get_current_user

db_client = get_db_client()
logger = get_logger()

user = get_current_user()

st.title("Run the pipeline")

_PIPELINE_RUNNING_KEY = "pipeline_running"
_PIPELINE_PENDING_KEY = "pipeline_pending"
_PIPELINE_ACCOUNT_KEY = "pipeline_account"
_PIPELINE_LOGS_KEY = "pipeline_logs"
_PIPELINE_OUTCOME_KEY = "pipeline_outcome"

st.session_state.setdefault(_PIPELINE_RUNNING_KEY, False)
st.session_state.setdefault(_PIPELINE_PENDING_KEY, False)
st.session_state.setdefault(_PIPELINE_LOGS_KEY, [])
st.session_state.setdefault(_PIPELINE_OUTCOME_KEY, None)


def _render_log_panel(placeholder: DeltaGenerator, lines: list[str]) -> None:
    rendered_lines = "\n".join(html.escape(line) for line in lines[-80:])
    placeholder.markdown(
        f"""
        <div style="
            border: 1px solid rgba(128, 128, 128, 0.35);
            border-radius: 0.5rem;
            background: rgba(128, 128, 128, 0.08);
            height: 260px;
            overflow-y: auto;
            padding: 0.75rem;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.85rem;
            line-height: 1.35;
            white-space: pre-wrap;
        ">{rendered_lines or "Waiting for logs..."}</div>
        """,
        unsafe_allow_html=True,
    )


class _StreamlitLogHandler(logging.Handler):
    """Render log records into a Streamlit placeholder during a pipeline run."""

    def __init__(self, placeholder: DeltaGenerator, lines: list[str]):
        super().__init__(level=logging.INFO)
        self.placeholder = placeholder
        self.lines = lines
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))
        _render_log_panel(self.placeholder, self.lines)


def _format_account(account: Account) -> str:
    return f"{account.platform.value} - {account.username}"


with st.form("pipeline_form"):
    accounts = get_accounts_for_user(user, db_client)
    account = st.selectbox(
        "Account",
        options=accounts,
        format_func=_format_account,
        disabled=not accounts or st.session_state[_PIPELINE_RUNNING_KEY],
    )
    submitted = st.form_submit_button(
        "Run pipeline",
        disabled=not accounts or st.session_state[_PIPELINE_RUNNING_KEY],
    )

if not accounts:
    st.info("There are no platform accounts linked.")

if submitted:
    st.session_state[_PIPELINE_ACCOUNT_KEY] = account
    st.session_state[_PIPELINE_PENDING_KEY] = True
    st.session_state[_PIPELINE_RUNNING_KEY] = True
    st.session_state[_PIPELINE_LOGS_KEY] = []
    st.session_state[_PIPELINE_OUTCOME_KEY] = None
    st.rerun()

if st.session_state[_PIPELINE_PENDING_KEY]:
    root_logger = logging.getLogger()
    log_placeholder = st.empty()
    _render_log_panel(log_placeholder, st.session_state[_PIPELINE_LOGS_KEY])
    streamlit_handler = _StreamlitLogHandler(
        log_placeholder,
        st.session_state[_PIPELINE_LOGS_KEY],
    )
    root_logger.addHandler(streamlit_handler)

    with st.status("Pipeline is running...", expanded=True) as status:
        started_at = datetime.now()
        try:
            result = run_pipeline(user, st.session_state[_PIPELINE_ACCOUNT_KEY])
        except Exception as e:
            status.update(label="Pipeline failed.", state="error", expanded=True)
            st.session_state[_PIPELINE_OUTCOME_KEY] = ("error", f"Pipeline failed: {e}")
            logger.exception("Pipeline failed from Streamlit page.")
        else:
            duration = (datetime.now() - started_at).total_seconds()
            status.update(label="Pipeline completed.", state="complete", expanded=True)
            st.session_state[_PIPELINE_OUTCOME_KEY] = (
                "success",
                f"Pipeline finished with result '{result.result.value}' in {duration:.2f}s.",
            )
        finally:
            root_logger.removeHandler(streamlit_handler)
            st.session_state[_PIPELINE_PENDING_KEY] = False
            st.session_state[_PIPELINE_RUNNING_KEY] = False
            st.rerun()

if not st.session_state[_PIPELINE_RUNNING_KEY] and st.session_state[_PIPELINE_OUTCOME_KEY]:
    log_placeholder = st.empty()
    _render_log_panel(log_placeholder, st.session_state[_PIPELINE_LOGS_KEY])

    outcome_type, outcome_message = st.session_state[_PIPELINE_OUTCOME_KEY]
    if outcome_type == "success":
        st.success(outcome_message)
    else:
        st.error(outcome_message)
