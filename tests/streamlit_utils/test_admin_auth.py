"""Tests for Streamlit admin access helpers."""

from __future__ import annotations

import streamlit_utils.admin_auth as admin_auth
from chess_teacher.platform.user import User


def _user(email: str | None) -> User:
    return User(user_id="test-user", sub="sub", provider="google", email=email)


class TestIsAdmin:
    def test_admin_email_is_recognized(self, monkeypatch) -> None:
        monkeypatch.setattr(admin_auth, "ADMIN_EMAILS", frozenset({"admin@example.com"}))
        assert admin_auth.is_admin(_user("admin@example.com")) is True

    def test_admin_email_match_is_case_insensitive(self, monkeypatch) -> None:
        monkeypatch.setattr(admin_auth, "ADMIN_EMAILS", frozenset({"admin@example.com"}))
        assert admin_auth.is_admin(_user("Admin@Example.com")) is True

    def test_admin_list_entry_case_is_ignored(self, monkeypatch) -> None:
        monkeypatch.setattr(admin_auth, "ADMIN_EMAILS", frozenset({"Admin@Example.com"}))
        assert admin_auth.is_admin(_user("admin@example.com")) is True

    def test_non_admin_email_is_rejected(self, monkeypatch) -> None:
        monkeypatch.setattr(admin_auth, "ADMIN_EMAILS", frozenset({"admin@example.com"}))
        assert admin_auth.is_admin(_user("user@example.com")) is False

    def test_missing_email_is_rejected(self, monkeypatch) -> None:
        monkeypatch.setattr(admin_auth, "ADMIN_EMAILS", frozenset({"admin@example.com"}))
        assert admin_auth.is_admin(_user(None)) is False
