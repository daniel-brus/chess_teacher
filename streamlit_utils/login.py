import streamlit as st

from chess_teacher.platform.user import User
from chess_teacher.utils.db_client import get_db_client
from chess_teacher.utils.exception_utils import AuthError
from chess_teacher.utils.general_utils import get_current_datetime
from chess_teacher.utils.logging_utils import get_logger
from streamlit_utils.session_state import set_current_user, st_user_is_logged_in


class LoginScreen:
    """Handles user authentication and session management."""

    def __init__(self):
        self.logger = get_logger()
        self.db_client = get_db_client()

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
            result = User.from_st_user(user)
        except Exception as e:
            self.logger.log_and_raise(e)
        result.save_to_db(self.db_client)
        return result

    def _exists_in_db(self, user: dict) -> bool:
        """Check if the st.user (dict) is already registered based on the generated id."""
        id = User.generate_id(user)
        return User.exists_in_db(self.db_client, id)

    def _fetch_existing_user(self, *, id: str | None = None, user: dict = {}) -> User:
        """Fetch an existing User object from the database, using an id or st.user (dict)."""
        return User.fetch_from_db(self.db_client, id=id, user=user)

    def display(self):
        self.logger.info("Login screen started.")
        if not st_user_is_logged_in():
            st.header("Log in to app")
            if st.button("Log in with Google"):
                st.login("google")
            st.stop()
        else:
            if st.session_state.get("current_user", {}):
                return
            now = get_current_datetime()
            st_user = st.user.to_dict()
            if not self._exists_in_db(st_user):
                user = self._create_new_user(st_user)
            else:
                user = self._fetch_existing_user(user=st_user)

            user.upsert_latest(self.db_client, "latest_login", now)
            set_current_user(user)
