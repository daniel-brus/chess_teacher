"""Admin access checks for the Streamlit app."""

from __future__ import annotations

import streamlit as st

from chess_teacher.platform.user import User
from streamlit_utils.login import require_authenticated_user

# Hardcoded admin allowlist. Add lowercase emails; matching is case-insensitive.
ADMIN_EMAILS: frozenset[str] = frozenset({
    # "you@example.com",
    "danielbrus24@gmail.com"
})


def is_admin(user: User) -> bool:
    """Return whether ``user`` has admin rights based on ``ADMIN_EMAILS``."""
    admin_emails = {email.strip().lower() for email in ADMIN_EMAILS}
    if not admin_emails:
        return False

    candidate_emails: set[str] = set()
    if user.email:
        candidate_emails.add(user.email.strip().lower())

    try:
        if st.user.is_logged_in:
            oauth_email = st.user.get("email")
            if oauth_email:
                candidate_emails.add(str(oauth_email).strip().lower())
    except Exception:
        pass

    return bool(candidate_emails & admin_emails)


def require_admin_user() -> User:
    """
    Require a logged-in user with admin rights.

    Call at the top of admin-only page scripts (after ``configure_page``).
    """
    user = require_authenticated_user()
    if not is_admin(user):
        st.error("You do not have permission to access this page.")
        st.stop()
    return user
