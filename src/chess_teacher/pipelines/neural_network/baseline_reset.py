"""Reset baseline training bookkeeping for a cold restart."""

from __future__ import annotations

from dataclasses import dataclass, replace

from chess_teacher.pipelines.neural_network.models import (
    PROCESSED_FLAG_BASELINE,
    BaselineModel,
    BaselineModelStatus,
    TrainingState,
)
from chess_teacher.pipelines.neural_network.split_registry import get_split_registry
from chess_teacher.pipelines.neural_network.splits import DEFAULT_SPLIT_SALT
from chess_teacher.utils.db.client import DatabaseClient
from chess_teacher.utils.general_utils import quote_literal
from chess_teacher.utils.logging import get_logger

logger = get_logger()


@dataclass(frozen=True)
class BaselineTrainingResetResult:
    archived_versions: tuple[str, ...]
    models_archived: int
    flags_cleared: int


def reset_baseline_training(
    db_client: DatabaseClient,
    *,
    archive_models: bool = True,
    dry_run: bool = False,
    split_version: str = DEFAULT_SPLIT_SALT,
) -> BaselineTrainingResetResult:
    """Archive active baseline rows and reset baseline queue flags.

    Does not delete MLflow artifacts or DB rows - only sets ``status=archived``
    and NULLs ``already_processed_baseline`` for ``split_version``. Dry-run
    writes nothing. Personal flags stay.
    """
    db_client.ensure_metadata(TrainingState.get_metadata())
    db_client.ensure_metadata(BaselineModel.get_metadata())

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

    flags_cleared = 0
    if not dry_run:
        flags_cleared = get_split_registry(db_client, split_version=split_version).clear_processed(
            flag_column=PROCESSED_FLAG_BASELINE,
        )
        logger.info(
            "Cleared baseline processed flags split_version=%s updated=%s",
            split_version,
            flags_cleared,
        )

    return BaselineTrainingResetResult(
        archived_versions=tuple(archived_versions),
        models_archived=len(archived_versions),
        flags_cleared=flags_cleared,
    )
