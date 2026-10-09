"""Plan val-score rows for models trained before ``training_scores`` existed.

The ops script loads weights and writes the rows. This module only decides
which models still need a score and how that row is keyed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from chess_teacher.pipelines.neural_network.eval_metrics import EvalMetrics
from chess_teacher.pipelines.neural_network.models import BaselineModel, PersonalModel
from chess_teacher.pipelines.neural_network.training_scores import (
    TrainingScore,
    training_score_from_eval,
)

BACKFILL_RUN_PREFIX = "backfill:"
BASELINE_PIPELINE_NAME = "baseline_training"
PERSONAL_PIPELINE_NAME = "personal_training"


@dataclass(frozen=True)
class ScoreTarget:
    """One registered candidate-style model that has no score row yet."""

    model_id: str
    version: str
    user_id: str | None
    model_uri: str
    trained_at: datetime
    pipeline_name: str


def backfill_run_id(model_id: str) -> str:
    """Stable primary key so a second run does not invent another row."""
    cleaned = model_id.strip()
    if not cleaned:
        raise ValueError("model_id is required")
    return f"{BACKFILL_RUN_PREFIX}{cleaned}"


def select_score_targets(
    baselines: list[BaselineModel],
    personals: list[PersonalModel],
    scored_model_ids: set[str],
) -> list[ScoreTarget]:
    """Candidate-style models with weights and no existing score row.

    Platform models come first, then each user, oldest training date first.
    """
    targets: list[ScoreTarget] = []
    for baseline in baselines:
        target = _baseline_target(baseline, scored_model_ids)
        if target is not None:
            targets.append(target)
    for personal in personals:
        target = _personal_target(personal, scored_model_ids)
        if target is not None:
            targets.append(target)
    targets.sort(
        key=lambda target: (
            target.user_id is not None,
            target.user_id or "",
            target.trained_at,
            target.version,
        )
    )
    return targets


def apply_limit(targets: list[ScoreTarget], limit: int | None) -> list[ScoreTarget]:
    if limit is None:
        return list(targets)
    if limit < 1:
        raise ValueError("limit must be at least 1")
    return list(targets[:limit])


def group_targets(targets: list[ScoreTarget]) -> list[tuple[str | None, list[ScoreTarget]]]:
    """Group by exam. ``None`` is the platform val set. Order follows ``targets``."""
    groups: dict[str | None, list[ScoreTarget]] = {}
    order: list[str | None] = []
    for target in targets:
        key = target.user_id
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(target)
    return [(key, groups[key]) for key in order]


def iter_batches(targets: list[ScoreTarget], batch_size: int) -> list[list[ScoreTarget]]:
    """Split one exam's models so only ``batch_size`` Keras models are loaded at once."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    return [targets[start : start + batch_size] for start in range(0, len(targets), batch_size)]


def training_score_for_backfill(target: ScoreTarget, metrics: EvalMetrics) -> TrainingScore:
    """Score row dated at the model's training time, not the backfill clock."""
    return training_score_from_eval(
        run_id=backfill_run_id(target.model_id),
        pipeline_name=target.pipeline_name,
        user_id=target.user_id,
        version=target.version,
        model_id=target.model_id,
        scored_at=target.trained_at,
        metrics=metrics,
    )


def _baseline_target(model: BaselineModel, scored_model_ids: set[str]) -> ScoreTarget | None:
    uri = model.model_uri
    if model.id in scored_model_ids or not uri or not model.looks_like_candidate_style():
        return None
    return ScoreTarget(
        model_id=model.id,
        version=model.version,
        user_id=None,
        model_uri=uri,
        trained_at=model.trained_at,
        pipeline_name=BASELINE_PIPELINE_NAME,
    )


def _personal_target(model: PersonalModel, scored_model_ids: set[str]) -> ScoreTarget | None:
    uri = model.model_uri
    if model.id in scored_model_ids or not uri or not model.looks_like_candidate_style():
        return None
    return ScoreTarget(
        model_id=model.id,
        version=model.version,
        user_id=model.user_id,
        model_uri=uri,
        trained_at=model.trained_at,
        pipeline_name=PERSONAL_PIPELINE_NAME,
    )
