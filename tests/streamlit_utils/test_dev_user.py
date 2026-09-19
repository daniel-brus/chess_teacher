"""Tests for secret-gated DEV environment Streamlit login."""

from __future__ import annotations

from unittest.mock import MagicMock

import streamlit_utils.login as login
from chess_teacher.platform.user import User
from streamlit_utils.dev_user import DEV_ENVIRONMENT_ST_USER, is_dev_environment_user_enabled


def test_dev_user_is_not_an_admin_email() -> None:
    from streamlit_utils.admin_auth import ADMIN_EMAILS

    email = str(DEV_ENVIRONMENT_ST_USER["email"]).strip().lower()
    assert email not in {item.strip().lower() for item in ADMIN_EMAILS}


def test_enabled_only_with_allowed_env_and_long_secret(monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "DEV")
    monkeypatch.setenv("STREAMLIT_DEV_USER_SECRET", "x" * 16)
    assert is_dev_environment_user_enabled() is True


def test_disabled_when_secret_missing(monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "DEV")
    monkeypatch.delenv("STREAMLIT_DEV_USER_SECRET", raising=False)
    assert is_dev_environment_user_enabled() is False


def test_disabled_when_secret_too_short(monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "DEV")
    monkeypatch.setenv("STREAMLIT_DEV_USER_SECRET", "short-secret")
    assert is_dev_environment_user_enabled() is False


def test_disabled_in_prod_even_with_secret(monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "PROD")
    monkeypatch.setenv("STREAMLIT_DEV_USER_SECRET", "x" * 16)
    assert is_dev_environment_user_enabled() is False


def test_disabled_in_agent_even_with_secret(monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "AGENT")
    monkeypatch.setenv("STREAMLIT_DEV_USER_SECRET", "x" * 16)
    assert is_dev_environment_user_enabled() is False


def test_disabled_when_environment_unset(monkeypatch) -> None:
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.setenv("STREAMLIT_DEV_USER_SECRET", "x" * 16)
    assert is_dev_environment_user_enabled() is False


def test_login_uses_dev_user_without_google(monkeypatch) -> None:
    monkeypatch.setattr(login, "is_dev_environment_user_enabled", lambda: True)
    db = MagicMock()
    monkeypatch.setattr(login, "get_db_client", lambda: db)
    assigned: dict[str, User] = {}
    monkeypatch.setattr(login, "set_current_user", lambda user: assigned.setdefault("user", user))
    monkeypatch.setattr(login.st, "session_state", {})

    screen = login.LoginScreen()
    monkeypatch.setattr(screen, "_exists_in_db", lambda _st_user: False)
    monkeypatch.setattr(User, "save_new_to_db", lambda self, _db: True)
    monkeypatch.setattr(User, "upsert_latest", lambda self, *_args, **_kwargs: None)

    screen.display()

    user = assigned["user"]
    assert user.provider == "dev"
    assert user.email == "dev-user@localhost"
    assert user.sub == DEV_ENVIRONMENT_ST_USER["sub"]
