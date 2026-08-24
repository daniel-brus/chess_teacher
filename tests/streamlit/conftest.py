"""Shared fixtures for Streamlit page AppTest smoke checks."""

from __future__ import annotations

from unittest.mock import MagicMock

import polars as pl
import pytest

from chess_teacher.platform.user import User


@pytest.fixture
def smoke_user() -> User:
    return User(
        user_id="smoke-user-id",
        sub="smoke-sub",
        provider="google",
        email="smoke@example.com",
        name="Smoke Tester",
    )


@pytest.fixture
def mock_db_client() -> MagicMock:
    return MagicMock(name="db_client")


@pytest.fixture
def patch_streamlit_page_deps(
    monkeypatch: pytest.MonkeyPatch,
    smoke_user: User,
    mock_db_client: MagicMock,
) -> User:
    """Patch auth, DB, logging, and common I/O so page scripts can render."""
    monkeypatch.setattr(
        "streamlit_utils.login.require_authenticated_user",
        lambda: smoke_user,
    )
    monkeypatch.setattr(
        "streamlit_utils.admin_auth.require_admin_user",
        lambda: smoke_user,
    )
    monkeypatch.setattr("streamlit_utils.page_logging.log_page_view", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "streamlit_utils.page_logging.log_user_action", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "chess_teacher.utils.db.client.get_db_client",
        lambda: mock_db_client,
    )
    monkeypatch.setattr(smoke_user, "get_linked_accounts", lambda _db: [])
    monkeypatch.setattr(smoke_user, "get_latest_pipeline_run", lambda _db: None)

    monkeypatch.setattr(
        "chess_teacher.platform.profile_picture.profile_pictures.app_logo_picture_ref",
        lambda variant: f"asset:app-logo-{variant}.svg",
    )
    monkeypatch.setattr(
        "chess_teacher.platform.profile_picture.profile_pictures.picture_img_src",
        lambda _picture: None,
    )
    monkeypatch.setattr(
        "chess_teacher.platform.profile_picture.profile_pictures.is_remote",
        lambda _picture: False,
    )
    monkeypatch.setattr("streamlit_utils.profile_ui.render_avatar_preview", lambda *a, **k: None)
    monkeypatch.setattr(
        "streamlit_utils.profile_ui.render_logo_preset_preview",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "chess_teacher.utils.object_storage.images.storage_image_data_uri",
        lambda *_a, **_k: None,
    )

    empty = pl.DataFrame()
    monkeypatch.setattr(
        "chess_teacher.maintenance.log_analytics.load_log_level_hourly_counts",
        lambda _db: empty,
    )
    monkeypatch.setattr(
        "chess_teacher.maintenance.log_analytics.load_exception_hourly_counts",
        lambda _db: empty,
    )
    return smoke_user
