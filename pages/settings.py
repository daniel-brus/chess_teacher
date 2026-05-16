import streamlit as st

from chess_teacher.utils.db_client import get_db_client
from streamlit_utils.session_state import force_logout, get_current_user

user = get_current_user()

db_client = get_db_client()

st.title("Personal Settings")


@st.dialog("Are you sure?")
def safe_rm_account():
    st.warning("Your user info will be lost forever")
    if st.button("I'm sure"):
        user.delete_from_db(db_client)
        force_logout()


if st.button("Remove account"):
    safe_rm_account()
