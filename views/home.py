import streamlit as st

from streamlit_utils.login import require_authenticated_user

user = require_authenticated_user()

st.title(f"Welcome to the Chess Teacher app, {user.name}!")

st.markdown("todo")
