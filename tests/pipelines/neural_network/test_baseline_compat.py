"""BaselineModel candidate_style compatibility (feat-dim gate)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from chess_teacher.pipelines.neural_network.candidate_eval import (
    MAX_CANDIDATES,
    MOVE_FEAT_DIM,
)
from chess_teacher.pipelines.neural_network.models import (
    BaselineModel,
    BaselineModelStatus,
)


def _model(metrics: dict[str, object]) -> BaselineModel:
    return BaselineModel(
        id="baseline-test",
        version="v99",
        status=BaselineModelStatus.CANDIDATE,
        trained_at=datetime.now(UTC),
        model_uri="s3://bucket/model.keras",
        eval_metrics=json.dumps(metrics),
    )


def test_looks_like_candidate_style_accepts_current_feat_dim() -> None:
    row = _model({
        "head_candidate_style": 1.0,
        "move_feat_dim": float(MOVE_FEAT_DIM),
        "max_candidates": float(MAX_CANDIDATES),
    })
    assert row.looks_like_candidate_style() is True


def test_looks_like_candidate_style_rejects_old_feat_dim() -> None:
    row = _model({
        "head_candidate_style": 1.0,
        "move_feat_dim": 22.0,  # v2-era
        "max_candidates": float(MAX_CANDIDATES),
    })
    assert row.looks_like_candidate_style() is False


def test_looks_like_candidate_style_rejects_legacy_without_feat_dim() -> None:
    row = _model({"head_candidate_style": 1.0, "max_candidates": float(MAX_CANDIDATES)})
    assert row.looks_like_candidate_style() is False


def test_looks_like_candidate_style_false_when_metrics_missing() -> None:
    row = BaselineModel(
        id="baseline-empty",
        version="v1",
        status=BaselineModelStatus.PRODUCTION,
        trained_at=datetime.now(UTC),
        model_uri=None,
        eval_metrics=None,
    )
    assert row.looks_like_candidate_style() is False
