"""Which registered models a training-score backfill will write."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from chess_teacher.pipelines.neural_network.candidate_eval import MAX_CANDIDATES, MOVE_FEAT_DIM
from chess_teacher.pipelines.neural_network.eval_metrics import EvalMetrics
from chess_teacher.pipelines.neural_network.models import BaselineModel, PersonalModel
from chess_teacher.pipelines.neural_network.training_score_backfill import (
    apply_limit,
    backfill_run_id,
    group_targets,
    iter_batches,
    select_score_targets,
    training_score_for_backfill,
)
from scripts.ops import backfill_training_scores


def _when(day: int) -> datetime:
    return datetime(2026, 9, day, tzinfo=UTC)


def _style() -> str:
    return json.dumps({
        "head_candidate_style": 1.0,
        "move_feat_dim": MOVE_FEAT_DIM,
        "max_candidates": MAX_CANDIDATES,
    })


def _baseline(
    version: str,
    *,
    day: int,
    uri: str | None = "s3://models/base",
    metrics: str | None = None,
) -> BaselineModel:
    return BaselineModel(
        id=f"base-{version}",
        version=version,
        trained_at=_when(day),
        model_uri=uri,
        eval_metrics=_style() if metrics is None else metrics,
    )


def _personal(version: str, *, day: int, user_id: str = "user-1") -> PersonalModel:
    return PersonalModel(
        id=f"{user_id}-{version}",
        user_id=user_id,
        version=version,
        trained_at=_when(day),
        model_uri="s3://models/personal",
        eval_metrics=_style(),
    )


def test_select_skips_scored_incompatible_and_missing_weights() -> None:
    ready = _baseline("v2", day=2)
    already = _baseline("v1", day=1)
    legacy = _baseline("v0", day=3, metrics="{}")
    missing = _baseline("v3", day=4, uri=None)
    mine = _personal("v1", day=5)
    targets = select_score_targets(
        [already, ready, legacy, missing],
        [mine],
        scored_model_ids={already.id},
    )
    assert [(target.model_id, target.user_id, target.pipeline_name) for target in targets] == [
        (ready.id, None, "baseline_training"),
        (mine.id, "user-1", "personal_training"),
    ]


def test_order_is_platform_then_user_oldest_first() -> None:
    later_base = _baseline("v2", day=8)
    early_base = _baseline("v1", day=1)
    later_user = _personal("v2", day=9, user_id="user-b")
    early_user = _personal("v1", day=3, user_id="user-a")
    targets = select_score_targets([later_base, early_base], [later_user, early_user], set())
    assert [target.model_id for target in targets] == [
        early_base.id,
        later_base.id,
        early_user.id,
        later_user.id,
    ]
    groups = group_targets(apply_limit(targets, 3))
    assert [key for key, _group in groups] == [None, "user-a"]
    assert [len(group) for _key, group in groups] == [2, 1]


def test_batches_and_limit_reject_zero() -> None:
    targets = select_score_targets([_baseline("v1", day=1)], [], set())
    assert [len(batch) for batch in iter_batches(targets * 3, 2)] == [2, 1]
    with pytest.raises(ValueError, match="batch_size"):
        iter_batches(targets, 0)
    with pytest.raises(ValueError, match="limit"):
        apply_limit(targets, 0)


def test_backfill_row_uses_training_time_and_stable_run_id() -> None:
    target = select_score_targets([_baseline("v4", day=4)], [], set())[0]
    metrics = EvalMetrics(
        top1_overall=0.41,
        top3_overall=0.72,
        top1_sf_agree=0.8,
        top3_sf_agree=0.9,
        top1_sf_disagree=0.2,
        top3_sf_disagree=0.5,
        top1_overall_weighted=0.39,
        n_eval=12,
        n_dropped=1,
        n_sf_agree=5,
        n_sf_disagree=7,
        sf_disagree_frac=0.58,
        sf_delta_mean_pawns=-0.33,
        sf_delta_median_pawns=-0.15,
    )
    row = training_score_for_backfill(target, metrics)
    assert row.run_id == backfill_run_id(target.model_id) == f"backfill:{target.model_id}"
    assert row.scored_at == target.trained_at
    assert row.val_top1 == 0.41
    assert row.sf_delta_mean_pawns == -0.33
    assert row.user_id is None
    with pytest.raises(ValueError, match="model_id"):
        backfill_run_id("  ")


def test_dry_run_prints_pending_without_loading_weights(monkeypatch, capsys) -> None:
    pending = _baseline("v7", day=7)
    monkeypatch.setattr(backfill_training_scores, "get_db_client", lambda: object())
    monkeypatch.setattr(
        backfill_training_scores.BaselineModel,
        "fetch_all_from_db",
        lambda _db: [pending],
    )
    monkeypatch.setattr(
        backfill_training_scores.PersonalModel,
        "fetch_all_from_db",
        lambda _db: [],
    )
    monkeypatch.setattr(
        backfill_training_scores.TrainingScore,
        "fetch_all_from_db",
        lambda _db: [],
    )
    monkeypatch.setattr("sys.argv", ["backfill_training_scores.py", "--dry-run"])
    assert backfill_training_scores.main() == 0
    output = capsys.readouterr().out
    assert "pending_count=1" in output
    assert "version=v7" in output
    assert "dry_run=true" in output
