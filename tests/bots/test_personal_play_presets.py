"""Personal play presets: promoted rows, platform fallback, and owner check."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from chess_teacher.bots import OpponentCategory
from chess_teacher.bots.presets import (
    category_for_preset_key,
    get_bot_preset,
    list_personal_play_presets,
    parse_personal_preset_key,
    personal_preset_key,
    reset_baseline_presets_cache_for_tests,
)
from chess_teacher.pipelines.neural_network.candidate_eval import (
    MAX_CANDIDATES,
    MOVE_FEAT_DIM,
)
from chess_teacher.pipelines.neural_network.models import (
    BaselineModel,
    BaselineModelStatus,
    PersonalModel,
)

USER = "user-a"


@pytest.fixture(autouse=True)
def _clear_preset_cache() -> None:
    reset_baseline_presets_cache_for_tests()
    yield
    reset_baseline_presets_cache_for_tests()


def _metrics(*, feat_dim: float | None = None) -> str:
    payload: dict[str, object] = {
        "head_candidate_style": 1.0,
        "max_candidates": float(MAX_CANDIDATES),
    }
    if feat_dim is not None:
        payload["move_feat_dim"] = feat_dim
    return json.dumps(payload)


def _personal(
    *,
    version: str,
    status: BaselineModelStatus,
    user_id: str = USER,
    feat_dim: float | None = None,
) -> PersonalModel:
    return PersonalModel(
        id=f"id-{user_id}-{version}",
        user_id=user_id,
        version=version,
        status=status,
        trained_at=datetime(2026, 9, 1, tzinfo=UTC),
        model_uri=f"s3://models/{user_id}/{version}.keras",
        eval_metrics=_metrics(feat_dim=feat_dim if feat_dim is not None else float(MOVE_FEAT_DIM)),
    )


def _baseline(*, version: str = "v9") -> BaselineModel:
    return BaselineModel(
        id=f"base-{version}",
        version=version,
        status=BaselineModelStatus.PRODUCTION,
        trained_at=datetime(2026, 9, 1, tzinfo=UTC),
        model_uri=f"s3://models/baseline/{version}.keras",
        eval_metrics=_metrics(feat_dim=float(MOVE_FEAT_DIM)),
    )


def test_personal_presets_skip_candidates_and_incompatible_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _personal(version="v1", status=BaselineModelStatus.PRODUCTION),
        _personal(version="v0", status=BaselineModelStatus.ARCHIVED),
        _personal(version="v_cand", status=BaselineModelStatus.CANDIDATE),
        _personal(version="v_old", status=BaselineModelStatus.PRODUCTION, feat_dim=22.0),
    ]
    monkeypatch.setattr(PersonalModel, "rows_for_user", classmethod(lambda cls, db, user: rows))
    presets, using_fallback = list_personal_play_presets(MagicMock(), USER)
    assert using_fallback is False
    assert [preset.key for preset in presets] == [
        personal_preset_key(USER, "v1"),
        personal_preset_key(USER, "v0"),
    ]
    assert category_for_preset_key(presets[0].key) == OpponentCategory.PERSONAL


def test_personal_category_falls_back_to_platform_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        PersonalModel,
        "rows_for_user",
        classmethod(lambda cls, db, user: []),
    )
    monkeypatch.setattr(
        BaselineModel,
        "latest_with_status",
        classmethod(lambda cls, db, status: _baseline()),
    )
    presets, using_fallback = list_personal_play_presets(MagicMock(), USER)
    assert using_fallback is True
    assert len(presets) == 1
    assert presets[0].key == "baseline:v9"
    assert "No personal model" in presets[0].description
    assert category_for_preset_key(presets[0].key) == OpponentCategory.BASELINE


def test_get_bot_preset_rejects_another_users_personal_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _personal(version="v1", status=BaselineModelStatus.PRODUCTION)
    monkeypatch.setattr(
        PersonalModel,
        "rows_for_user",
        classmethod(lambda cls, db, user: [row] if user == USER else []),
    )
    key = personal_preset_key(USER, "v1")
    assert get_bot_preset(key, db_client=MagicMock(), user_id=USER).label == "Personal v1"
    with pytest.raises(KeyError, match="not for this user"):
        get_bot_preset(key, db_client=MagicMock(), user_id="someone-else")


def test_personal_factory_names_the_bot(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeBot:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        "chess_teacher.bots.neural_baseline_bot.NeuralBaselineBot",
        _FakeBot,
    )
    row = _personal(version="v3", status=BaselineModelStatus.PRODUCTION)
    monkeypatch.setattr(
        PersonalModel,
        "rows_for_user",
        classmethod(lambda cls, db, user: [row]),
    )
    preset = get_bot_preset(personal_preset_key(USER, "v3"), db_client=MagicMock(), user_id=USER)
    preset.factory(temperature=0.4)
    assert captured["display_name"] == "Personal v3"
    assert captured["temperature"] == pytest.approx(0.4)


def test_parse_personal_preset_key_rejects_a_short_key() -> None:
    with pytest.raises(KeyError):
        parse_personal_preset_key("personal:onlyone")
