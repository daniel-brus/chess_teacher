import streamlit as st

from streamlit_utils.layout import set_bg
from streamlit_utils.login import require_authenticated_user

user = require_authenticated_user()
set_bg("#ffffff", "#e0e0e0")
st.title(f"Welcome to the Chess Teacher app, {user.name}!")

st.markdown("todo")
