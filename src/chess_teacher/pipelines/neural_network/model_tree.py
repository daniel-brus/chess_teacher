"""Start a new training tree after an architecture change.

Hybrid training already cold-starts weights when the parent file is the flat
baseline, but promotion still scores the new candidate against that older
production row. The older row has had more training rounds, so the comparison
is not a fair start for the new architecture.

This operation is a deliberate ops step. It is not run from the trainer: a
failed weight load must not archive production on its own. It archives every
baseline and personal row that is not the current hybrid scheme, clears the
parent-baseline pointer when that row is one of them, resets both training
queues, and clears ``personal_queue_baseline``. Archived models stay playable.
MLflow artifacts are not deleted.

Run it once when the served architecture changes. A later run archives nothing
new, but it does replay the queues again.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

from chess_teacher.pipelines.neural_network.models import (
    BaselineModel,
    BaselineModelStatus,
    PersonalModel,
    TrainingState,
)
from chess_teacher.pipelines.neural_network.training_queue_reset import reset_training_queues
from chess_teacher.utils.db.client import DatabaseClient
from chess_teacher.utils.logging import get_logger

logger = get_logger()

_HYBRID_METRIC = "encoder_hybrid_board"


def eval_metrics_are_hybrid_board(eval_metrics: str | None) -> bool:
    """True when a metrics blob was written by the hybrid board+state trainer."""
    if not eval_metrics:
        return False
    try:
        blob = json.loads(eval_metrics)
    except (TypeError, json.JSONDecodeError):
        return False
    if not isinstance(blob, dict):
        return False
    flag = blob.get(_HYBRID_METRIC)
    try:
        return float(flag) == 1.0
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class ModelTreeRowChange:
    """One row removed from parent selection."""

    kind: str
    version: str
    user_id: str | None
    previous_status: str
    archived: bool
    cleared_parent_baseline: bool


@dataclass(frozen=True)
class ModelTreeReset:
    """What a model-tree reset will do, or did."""

    changes: tuple[ModelTreeRowChange, ...]
    queue_rows: int
    cleared_scopes: tuple[str, ...]


def format_model_tree_reset(result: ModelTreeReset, *, applied: bool) -> str:
    """Plain-text report for the ops script."""
    archived = [change for change in result.changes if change.archived]
    baselines = [change for change in archived if change.kind == "baseline"]
    personals = [change for change in archived if change.kind == "personal"]
    archive_verb = "Archived" if applied else "Would archive"
    lines = [
        f"{archive_verb} {len(baselines)} baseline model(s) and {len(personals)} personal model(s)."
    ]
    for change in result.changes:
        if change.kind == "baseline":
            notes = [change.previous_status]
            if change.cleared_parent_baseline:
                notes.append("clear parent baseline")
            if not change.archived:
                notes.append("already archived")
            lines.append(f"  baseline {change.version} ({', '.join(notes)})")
        else:
            lines.append(
                f"  personal user={change.user_id} {change.version} ({change.previous_status})"
            )
    queue_verb = "Reset" if applied else "Would reset"
    lines.append(f"{queue_verb} both training queues on {result.queue_rows} train rows.")
    if result.cleared_scopes:
        scope_verb = "Cleared" if applied else "Would clear"
        scopes = ", ".join(result.cleared_scopes)
        lines.append(f"{scope_verb} personal_queue_baseline on {scopes}.")
    else:
        lines.append("No personal_queue_baseline values to clear.")
    return "\n".join(lines)


def start_new_model_tree(
    db_client: DatabaseClient,
    *,
    dry_run: bool = False,
) -> ModelTreeReset:
    """Archive non-hybrid models and replay both training queues.

    Hybrid rows (``encoder_hybrid_board`` = 1 in eval metrics) stay in place,
    including a hybrid parent baseline. Every other non-archived row becomes
    archived. A non-hybrid parent-baseline flag is cleared even when that row
    was already archived, so the next hybrid cold start has no model to beat.
    """
    changes: list[ModelTreeRowChange] = []

    for row in BaselineModel.fetch_all_ordered(db_client):
        if eval_metrics_are_hybrid_board(row.eval_metrics):
            continue
        needs_archive = row.status != BaselineModelStatus.ARCHIVED
        needs_clear = row.is_parent_baseline
        if not needs_archive and not needs_clear:
            continue
        changes.append(
            ModelTreeRowChange(
                kind="baseline",
                version=row.version,
                user_id=None,
                previous_status=row.status.value,
                archived=needs_archive,
                cleared_parent_baseline=needs_clear,
            )
        )
        if dry_run:
            continue
        replace(
            row,
            status=BaselineModelStatus.ARCHIVED,
            is_parent_baseline=False,
        ).save_to_db(db_client)
        logger.info(
            "Model tree archived baseline version=%s previous_status=%s cleared_parent=%s",
            row.version,
            row.status.value,
            needs_clear,
        )

    for row in PersonalModel.fetch_all_from_db(db_client, order_by='"trained_at" DESC'):
        if eval_metrics_are_hybrid_board(row.eval_metrics):
            continue
        if row.status == BaselineModelStatus.ARCHIVED:
            continue
        changes.append(
            ModelTreeRowChange(
                kind="personal",
                version=row.version,
                user_id=row.user_id,
                previous_status=row.status.value,
                archived=True,
                cleared_parent_baseline=False,
            )
        )
        if dry_run:
            continue
        replace(row, status=BaselineModelStatus.ARCHIVED).save_to_db(db_client)
        logger.info(
            "Model tree archived personal user=%s version=%s previous_status=%s",
            row.user_id,
            row.version,
            row.status.value,
        )

    queue_rows = reset_training_queues(db_client, dry_run=dry_run)

    cleared_scopes: list[str] = []
    for state in TrainingState.fetch_all_from_db(db_client):
        if not state.personal_queue_baseline:
            continue
        cleared_scopes.append(state.scope)
        if dry_run:
            continue
        replace(state, personal_queue_baseline=None).save_to_db(db_client)
        logger.info("Cleared personal_queue_baseline scope=%s", state.scope)

    result = ModelTreeReset(
        changes=tuple(changes),
        queue_rows=queue_rows,
        cleared_scopes=tuple(cleared_scopes),
    )
    logger.info(
        "Model tree reset dry_run=%s archived=%s queue_rows=%s cleared_scopes=%s",
        dry_run,
        len(changes),
        queue_rows,
        len(cleared_scopes),
    )
    return result
