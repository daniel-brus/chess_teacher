"""Secret-gated development user for local and Cloud Agent Streamlit.

This bypasses Google OAuth only when ENVIRONMENT is an explicit non-production
value and STREAMLIT_DEV_USER_SECRET is set to a long secret. Presence of the
environment name alone is not enough.
"""

from __future__ import annotations

from typing import Any

from chess_teacher.utils.env_utils import get_environment, get_optional_env_variable

DEV_USER_ALLOWED_ENVIRONMENTS = frozenset({"DEV", "LOCAL", "TEST"})
DEV_USER_DENIED_ENVIRONMENTS = frozenset({"PROD", "AGENT"})
DEV_USER_SECRET_NAME = "STREAMLIT_DEV_USER_SECRET"
DEV_USER_SECRET_MIN_LENGTH = 16

DEV_ENVIRONMENT_ST_USER: dict[str, Any] = {
    "sub": "chess-teacher-dev-environment-user",
    "provider": "dev",
    "email": "dev-user@localhost",
    "name": "Dev User",
    "email_verified": True,
}


def is_dev_environment_user_enabled() -> bool:
    """Return whether Streamlit may authenticate as the fixed DEV user."""
    environment = (get_environment() or "").strip().upper()
    if not environment or environment in DEV_USER_DENIED_ENVIRONMENTS:
        return False
    if environment not in DEV_USER_ALLOWED_ENVIRONMENTS:
        return False
    secret = get_optional_env_variable(DEV_USER_SECRET_NAME)
    return len(secret) >= DEV_USER_SECRET_MIN_LENGTH
