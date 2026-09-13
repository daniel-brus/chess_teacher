import streamlit as st

from streamlit_utils.legal import TERMS_OF_USE_MARKDOWN, render_legal_page_nav
from streamlit_utils.page_config import LEGAL_UPDATED_ON, configure_page

configure_page("Terms of use")
render_legal_page_nav()

st.title("Terms of use")
st.caption(f"Last updated: {LEGAL_UPDATED_ON}")
st.markdown(TERMS_OF_USE_MARKDOWN)
