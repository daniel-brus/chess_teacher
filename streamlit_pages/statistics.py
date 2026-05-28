import streamlit as st

from streamlit_utils.login import require_authenticated_user

user = require_authenticated_user()

st.title("Statistics page")

st.text("Not yet implemented")
