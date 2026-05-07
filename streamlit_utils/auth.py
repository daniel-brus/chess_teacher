import streamlit as st

from chess_teacher.utils.exceptions import AuthError


def login_screen():
    st.header("Log in to app")
    if st.button("Log in with Google"):
        st.login()
    user_info = extract_user_info(st.user)
    # TODO: load necessary info into database
    # TODO: load necessary info into session state: just name and sub?
    return user_info


def extract_user_info(user):
    try:
        if user.get("aud", None) != st.secrets["client_id"]:
            raise AuthError("Invalid audience in user info.")
        return {
            "sub": user.get("sub"),  # unique ID, no fallback
            "email": user.get("email", None),
            "name": user.get("name", None),
            "picture": user.get("picture", None),
            "given_name": user.get("given_name", None),
            "family_name": user.get("family_name", None),
            "provider": user.get("provider", None),
            "email_verified": user.get("email_verified", None),
        }
    except AuthError as e:
        st.error("Error occurred while extracting user information.")
        raise e
