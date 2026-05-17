import streamlit as st

from streamlit_utils.session_state import get_current_user

user = get_current_user()

st.title(f"Welcome to the Chess Teacher app, {user.name}!")

st.markdown("todo")
