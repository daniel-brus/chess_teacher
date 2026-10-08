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
    assert any("Welcome to Chess Teacher" in title for title in _title_values(at))
    assert any("Smoke Tester" in title for title in _title_values(at))
    assert any(
        "linking a chess.com or lichess account" in info.lower() for info in _info_values(at)
    )


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


def test_training_page_renders_empty_state(
    patch_streamlit_page_deps: User,
) -> None:
    at = _run_page("training.py")
    assert "Training" in _title_values(at)
    assert any("no training models yet" in info.lower() for info in _info_values(at))


def test_training_page_renders_scores_and_lineage(
    patch_streamlit_page_deps: User,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import UTC, datetime

    from chess_teacher.pipelines.neural_network.models import BaselineModel, BaselineModelStatus
    from chess_teacher.pipelines.neural_network.training_progress import TrainingProgress
    from chess_teacher.pipelines.neural_network.training_scores import training_score_from_eval
    from tests.pipelines.neural_network.test_training_progress import _metrics

    trained_at = datetime(2026, 10, 2, tzinfo=UTC)
    baseline = BaselineModel(
        id="base-v1",
        version="v1",
        trained_at=trained_at,
        status=BaselineModelStatus.PRODUCTION,
        is_parent_baseline=True,
    )
    score = training_score_from_eval(
        run_id="run-base-v1",
        pipeline_name="baseline_training",
        user_id=None,
        version="v1",
        model_id=baseline.id,
        scored_at=trained_at,
        metrics=_metrics(disagree=0.2),
    )
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.training_progress.load_training_progress",
        lambda _db, _user_id: TrainingProgress(
            baselines=(baseline,),
            personals=(),
            scores=(score,),
        ),
    )
    at = _run_page("training.py")
    assert "Training" in _title_values(at)
    assert any(str(box.label) == "Metrics" for box in at.multiselect)
    assert at.dataframe
    captions = [str(caption.value) for caption in at.caption]
    assert any("personal model" in caption for caption in captions)


def test_privacy_page_renders() -> None:
    at = _run_page("privacy.py")
    assert "Privacy policy" in _title_values(at)
    markdown = "\n".join(_markdown_values(at))
    assert "strictly necessary" in markdown.lower()
    assert "google" in markdown.lower()


def test_terms_page_renders() -> None:
    at = _run_page("terms.py")
    assert "Terms of use" in _title_values(at)
    markdown = "\n".join(_markdown_values(at))
    assert "as is" in markdown.lower()
    assert "google" in markdown.lower()
