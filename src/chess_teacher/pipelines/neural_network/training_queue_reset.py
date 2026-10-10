"""Reset platform and personal training queue markers."""

from __future__ import annotations

from chess_teacher.pipelines.neural_network.split_registry import SplitRegistry
from chess_teacher.utils.db.client import DatabaseClient
from chess_teacher.utils.logging import get_logger

logger = get_logger()


def reset_training_queues(
    db_client: DatabaseClient,
    *,
    dry_run: bool = False,
) -> int:
    """Reset processed markers and retry counters for train rows in both queues.

    Model records, training cutoffs, and validation/test split assignments are
    preserved. Returns the number of affected train rows (or rows to reset in
    dry-run mode).
    """
    count = SplitRegistry(db_client).reset_training_queue_markers(dry_run=dry_run)
    logger.info("Training queue reset dry_run=%s train_rows=%s", dry_run, count)
    return count
