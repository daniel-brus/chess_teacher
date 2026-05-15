import streamlit as st

from chess_teacher.platform.user import User
from chess_teacher.utils.db_client import get_db_client
from chess_teacher.utils.exception_utils import AuthError
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
            self.logger.log_and_raise(AuthError("Failed to check login status"))

    def _create_new_user(self, user: dict) -> User:
        """Verify a user entry (st.user dict), create a User object and save
        the user to the database. If the user already exists, do nothing."""
        try:
            provider = user.get("provider", None)
            if not provider:
                self.logger.log_and_raise(AuthError("User missing 'provider' field"))
            client_id = st.secrets["auth"].get(provider, {}).get("client_id", None)
            if user.get("aud", None) != client_id:
                self.logger.log_and_raise(AuthError("Invalid user: audience mismatch"))
            if not user.get("email_verified", False):
                self.logger.warning(
                    f"User email not verified: {user.get("email", '"email not found')}"
                )
            result = User(user)
        except Exception as e:
            self.logger.log_and_raise(e)
        result.save_to_db(self.db_client)
        return result

    def _exists_in_db(self, user: dict) -> bool:
        """Check if the st.user is already registered based on the generated id."""
        id = User.generate_id(user)
        return User.exists_in_db(self.db_client, id)

    def _fetch_existing_user(self, *, id: str | None = None, user: dict = {}) -> User:
        """Fetch an existing User object from the database, using an id or st.user."""
        return User.fetch_from_db(self.db_client, id=id, user=user)

    def display(self):
        if not self._user_is_logged_in():
            st.header("Log in to app")
            if st.button("Log in with Google"):
                st.login("google")
        else:
            if not self._exists_in_db(st.user):
                user = self._create_new_user(st.user)
            else:
                user = self._fetch_existing_user(user=st.user)
            st.session_state["current_user"] = user

            st.write(f"Welcome, {st.user.get("name", "User")}!")
            if st.button("Log out"):
                st.logout()
                st.rerun()
