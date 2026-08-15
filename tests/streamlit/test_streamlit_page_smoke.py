"""Basic AppTest smoke coverage for each Streamlit page.

Pages are executed with auth/DB/storage patched. These checks assert the page
renders its shell (title / empty-state copy / setup widgets) without exercising
deep feature flows (chess moves, pipeline runs, chart data, etc.).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from chess_teacher.platform.user import User

pytestmark = pytest.mark.integration

_PAGES_DIR = Path(__file__).resolve().parents[2] / "streamlit_pages"


def _run_page(relative_name: str) -> AppTest:
    path = _PAGES_DIR / relative_name
    assert path.is_file(), f"missing page script: {path}"
    at = AppTest.from_file(str(path), default_timeout=15)
    at.run()
    assert not at.exception, f"{relative_name} raised: {at.exception}"
    return at


def _title_values(at: AppTest) -> list[str]:
    return [str(title.value) for title in at.title]


def _info_values(at: AppTest) -> list[str]:
    return [str(info.value) for info in at.info]


def _markdown_values(at: AppTest) -> list[str]:
    return [str(md.value) for md in at.markdown]


def test_home_page_renders_welcome(
    patch_streamlit_page_deps: User,
) -> None:
    at = _run_page("home.py")
    assert any("Welcome to the Chess Teacher app" in title for title in _title_values(at))
    assert any("Smoke Tester" in title for title in _title_values(at))
    assert any(md.strip() == "todo" for md in _markdown_values(at))


def test_pipeline_page_renders_empty_accounts_state(
    patch_streamlit_page_deps: User,
) -> None:
    at = _run_page("pipeline.py")
    assert "Run the pipeline" in _title_values(at)
    assert any("no platform accounts linked" in info.lower() for info in _info_values(at))
    assert at.button
    assert at.button[0].label == "Run pipeline"
    assert at.button[0].disabled


def test_play_page_renders_setup_shell(
    patch_streamlit_page_deps: User,
) -> None:
    at = _run_page("play.py")
    assert "Play a game of chess" in _title_values(at)
    assert any("Opponent type" in str(radio.label) for radio in at.radio)
    assert any(button.label == "Start game" for button in at.button)
    assert any("Your color" in str(box.label) for box in at.selectbox)


def test_statistics_page_renders_empty_accounts_state(
    patch_streamlit_page_deps: User,
) -> None:
    at = _run_page("statistics.py")
    assert "Game statistics" in _title_values(at)
    assert any("link a platform account" in info.lower() for info in _info_values(at))


def test_settings_page_renders_tabs(
    patch_streamlit_page_deps: User,
) -> None:
    at = _run_page("settings.py")
    assert "Personal Settings" in _title_values(at)
    tab_labels = [str(tab.label) for tab in at.tabs]
    for expected in (
        "Profile",
        "Schedule",
        "Platform accounts",
        "Appearance",
        "Delete account",
    ):
        assert expected in tab_labels


def test_admin_page_renders_empty_aggregates_state(
    patch_streamlit_page_deps: User,
) -> None:
    at = _run_page("admin.py")
    assert "Logging dashboard" in _title_values(at)
    assert any("no log aggregates yet" in info.lower() for info in _info_values(at))
