"""Reset baseline training bookkeeping for a cold restart."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime

from chess_teacher.pipelines.neural_network.models import (
    BASELINE_TRAINING_SCOPE,
    BaselineModel,
    BaselineModelStatus,
    TrainingState,
)
from chess_teacher.utils.db.client import DatabaseClient
from chess_teacher.utils.general_utils import quote_literal
from chess_teacher.utils.logging import get_logger

logger = get_logger()


@dataclass(frozen=True)
class BaselineTrainingResetResult:
    previous_cutoff: datetime | None
    archived_versions: tuple[str, ...]
    cutoff_cleared: bool
    models_archived: int


def reset_baseline_training(
    db_client: DatabaseClient,
    *,
    clear_cutoff: bool = True,
    archive_models: bool = True,
    dry_run: bool = False,
) -> BaselineTrainingResetResult:
    """Clear baseline data cutoff and archive active baseline model rows.

    Does not delete MLflow artifacts or DB rows — only sets ``status=archived``
    and ``last_trained_data_cutoff=NULL`` so the next train cold-starts.
    """
    db_client.ensure_metadata(TrainingState.get_metadata())
    db_client.ensure_metadata(BaselineModel.get_metadata())

    previous = TrainingState.for_baseline(db_client)
    archived_versions: list[str] = []

    if archive_models:
        active_where = f"status != {quote_literal(BaselineModelStatus.ARCHIVED.value)}"
        for row in BaselineModel.fetch_all_from_db(
            db_client,
            where=active_where,
            order_by='"version" ASC',
        ):
            archived_versions.append(row.version)
            if dry_run:
                continue
            replace(row, status=BaselineModelStatus.ARCHIVED).save_to_db(db_client)
            logger.info("Archived baseline version=%s (was %s)", row.version, row.status)

    cutoff_cleared = False
    if clear_cutoff:
        cutoff_cleared = True
        if not dry_run:
            TrainingState(
                scope=previous.scope or BASELINE_TRAINING_SCOPE,
                last_trained_data_cutoff=None,
                last_min_data_check_at=previous.last_min_data_check_at,
            ).save_to_db(db_client)
            logger.info(
                "Cleared baseline training cutoff (was %s)",
                previous.last_trained_data_cutoff,
            )

    return BaselineTrainingResetResult(
        previous_cutoff=previous.last_trained_data_cutoff,
        archived_versions=tuple(archived_versions),
        cutoff_cleared=cutoff_cleared,
        models_archived=len(archived_versions),
    )
