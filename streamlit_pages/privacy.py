import streamlit as st

from streamlit_utils.legal import PRIVACY_POLICY_MARKDOWN, render_legal_page_nav
from streamlit_utils.page_config import LEGAL_UPDATED_ON, configure_page

configure_page("Privacy policy")
render_legal_page_nav()

st.title("Privacy policy")
st.caption(f"Last updated: {LEGAL_UPDATED_ON}")
st.markdown(PRIVACY_POLICY_MARKDOWN)
