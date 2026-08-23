"""Play preset listing respects candidate_style feat-dim gate."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from chess_teacher.pipelines.neural_network.candidate_eval import (
    MAX_CANDIDATES,
    MOVE_FEAT_DIM,
)
from chess_teacher.pipelines.neural_network.models import (
    BaselineModel,
    BaselineModelStatus,
)
from chess_teacher.utils.chess_bots.presets import list_baseline_presets


def _row(*, version: str, feat_dim: float | None, status: BaselineModelStatus) -> BaselineModel:
    metrics: dict[str, object] = {
        "head_candidate_style": 1.0,
        "max_candidates": float(MAX_CANDIDATES),
    }
    if feat_dim is not None:
        metrics["move_feat_dim"] = feat_dim
    return BaselineModel(
        id=f"id-{version}",
        version=version,
        status=status,
        trained_at=datetime.now(UTC),
        model_uri=f"s3://models/{version}.keras",
        eval_metrics=json.dumps(metrics),
    )


def test_list_baseline_presets_skips_incompatible_feat_dim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _row(version="v_old", feat_dim=22.0, status=BaselineModelStatus.PRODUCTION),
        _row(
            version="v_ok",
            feat_dim=float(MOVE_FEAT_DIM),
            status=BaselineModelStatus.ARCHIVED,
        ),
        _row(
            version="v_cand",
            feat_dim=float(MOVE_FEAT_DIM),
            status=BaselineModelStatus.CANDIDATE,
        ),
    ]
    monkeypatch.setattr(
        BaselineModel,
        "fetch_all_ordered",
        classmethod(lambda cls, db: rows),
    )
    presets = list_baseline_presets(MagicMock())
    keys = {p.key for p in presets}
    assert "baseline:v_ok" in keys
    assert "baseline:v_old" not in keys
    assert "baseline:v_cand" not in keys  # candidates not playable on Play
