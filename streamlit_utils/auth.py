import streamlit as st

from chess_teacher.platform.user import User
from chess_teacher.utils.db_client import get_db_client
from chess_teacher.utils.exception_utils import AuthError
from chess_teacher.utils.general_utils import generate_hash
from chess_teacher.utils.logging_utils import get_logger


class LoginScreen:
    """Handles user authentication and session management."""

    def __init__(self):
        self.logger = get_logger()
        self.db_client = get_db_client()

    def _user_is_logged_in(self) -> bool:
        """Check if the user is logged in."""
        try:
            return st.user.is_logged_in
        except Exception:
            self.logger.log_and_raise(AuthError("Failed to check login status"), exc_info=True)

    def _extract_user(self, user) -> User:
        try:
            provider = user.get("provider", None)
            if not provider:
                self.logger.log_and_raise(AuthError("User missing 'provider' field"))
            client_id = st.secrets["auth"].get(provider, {}).get("client_id", None)
            if user.get("aud", None) != client_id:
                self.logger.log_and_raise(AuthError("Invalid user: audience mismatch"))
            result = {
                "id": generate_hash([user.get("sub"), provider]),  # hashed unique ID
                "sub": user.get("sub"),  # unique ID per provider, no fallback
                "email": user.get("email", None),
                "name": user.get("name", None),
                "picture": user.get("picture", None),
                "given_name": user.get("given_name", None),
                "family_name": user.get("family_name", None),
                "provider": provider,
                "email_verified": user.get("email_verified", False),
            }
            if not result["email_verified"]:
                self.logger.warning(f"User email not verified: {result["email"]}")
            return User(**result)
        except Exception as e:
            self.logger.log_and_raise(e)

    def display(self):
        if not self._user_is_logged_in():
            st.header("Log in to app")
            if st.button("Log in with Google"):
                st.login("google")
        else:
            st.write(f"Welcome, {st.user.get("name", "User")}!")
            if st.button("Log out"):
                st.logout()
                st.rerun()

            user = self._extract_user(st.user)
            user.save_to_db(self.db_client)
