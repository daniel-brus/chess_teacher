from datetime import datetime

import streamlit as st

from chess_teacher.ingestion.main import run_ingestion_pipeline
from chess_teacher.platform.account import Account
from chess_teacher.platform.users_accounts import get_accounts_for_user
from chess_teacher.utils.db_client import get_db_client
from chess_teacher.utils.logging_utils import get_logger
from streamlit_utils.login import require_authenticated_user
from streamlit_utils.progress_window import StreamlitProgressWindow

db_client = get_db_client()
logger = get_logger()

user = require_authenticated_user()

st.title("Run the pipeline")

_PIPELINE_PENDING_KEY = "pipeline_pending"
_PIPELINE_ACCOUNT_KEY = "pipeline_account"

st.session_state.setdefault(_PIPELINE_PENDING_KEY, False)


def _format_account(account: Account) -> str:
    return f"{account.platform.value} - {account.username}"


accounts = get_accounts_for_user(user, db_client)

with st.form("pipeline_form"):
    account = st.selectbox(
        "Account",
        options=accounts,
        format_func=_format_account,
        disabled=not accounts,
    )
    submitted = st.form_submit_button("Run pipeline", disabled=not accounts)

if not accounts:
    st.info("There are no platform accounts linked.")

if submitted:
    st.session_state[_PIPELINE_ACCOUNT_KEY] = account
    st.session_state[_PIPELINE_PENDING_KEY] = True
    st.rerun()

if st.session_state[_PIPELINE_PENDING_KEY]:
    with StreamlitProgressWindow() as progress:
        started_at = datetime.now()
        try:
            result = run_ingestion_pipeline(
                user.user_id,
                st.session_state[_PIPELINE_ACCOUNT_KEY],
                progress_window=progress,
            )
        except Exception:
            logger.error("Pipeline failed from Streamlit page.")
        finally:
            st.session_state[_PIPELINE_PENDING_KEY] = False
