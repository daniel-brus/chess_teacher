"""Assign persistent game splits during the per-account user pipeline."""

from __future__ import annotations

from chess_teacher.pipelines.neural_network.split_registry import SplitRegistry
from chess_teacher.pipelines.neural_network.splits import DEFAULT_SPLIT_SALT
from chess_teacher.utils.db.client import DatabaseClient
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.pipeline_utils.pipeline_base import PipelineContext, PipelineStep

logger = get_logger()


class AssignGameSplitsStep(PipelineStep):
    """Write ``ml.game_split_assignments`` for this account's eligible games.

    Same hash policy as ``backfill_game_splits.py``. Idempotent. Does **not**
    change training or promotion. New games can lag until the next user-pipeline
    run (typically ≤1 day).
    """

    def __init__(self, *, split_version: str = DEFAULT_SPLIT_SALT) -> None:
        super().__init__(name="AssignGameSplits")
        self.split_version = split_version

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        account_id = context.account_id
        if not account_id:
            raise ValueError("AssignGameSplits requires context.account_id")
        registry = SplitRegistry(db_client, split_version=self.split_version)
        result = registry.ensure_eligible_games_for_account(account_id)
        context.extras["split_assign"] = result
        logger.info(
            "AssignGameSplits split_version=%s account_id=%s eligible=%s new=%s already=%s",
            result.split_version,
            account_id,
            result.eligible_games,
            result.newly_assigned,
            result.already_assigned,
        )
