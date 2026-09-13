"""Train (+ optional promote) until eligible moves since cutoff are below the floor."""

from __future__ import annotations

import time

from chess_teacher.pipelines.neural_network.create_training_set import TrainingDataStore
from chess_teacher.pipelines.neural_network.main import (
    run_baseline_promotion_pipeline,
    run_baseline_training_pipeline,
)
from chess_teacher.pipelines.neural_network.models import TrainingState
from chess_teacher.pipelines.neural_network.pipeline_steps import (
    MAX_MOVES_PER_BASELINE_BATCH,
    MIN_NEW_MOVES_BASELINE,
)
from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.pipeline_utils.pipeline_helpers import (
    PipelineResult,
    PipelineRunResult,
)

logger = get_logger()


def _eligible_count() -> tuple[int, object]:
    db = get_db_client()
    state = TrainingState.for_baseline(db)
    n = TrainingDataStore(db).count_since(state.last_trained_data_cutoff)
    return n, state.last_trained_data_cutoff


def _run_ok(result: PipelineRunResult, *, label: str) -> bool:
    logger.info(
        "%s finished result=%s steps=%s",
        label,
        result.result.value,
        len(result.step_results),
    )
    for step in result.step_results:
        logger.info(
            "  %s: %s (%.1fs)%s",
            step.name,
            step.result.value,
            step.duration_seconds,
            f" err={step.error_message}" if step.error_message else "",
        )
    if result.result in {PipelineResult.FAILURE, PipelineResult.PARTIAL}:
        for msg in result.error_messages:
            logger.warning("%s pipeline error: %s", label, msg)
        return False
    return True


def loop_until_caught_up(*, promote: bool = True, max_rounds: int = 50) -> int:
    """Return 0 when caught up; non-zero on failure / stall / max_rounds.

    Stops when ``count_since(cutoff) < MIN_NEW_MOVES_BASELINE``. Remainder under
    that floor is left for later (same as a single skipped train job).
    """
    max_rounds = max(1, int(max_rounds))
    round_i = 0
    while round_i < max_rounds:
        n_left, cutoff = _eligible_count()
        logger.info(
            "Catch-up check round=%s eligible=%s cutoff=%s min=%s batch_cap=%s",
            round_i + 1,
            n_left,
            cutoff,
            MIN_NEW_MOVES_BASELINE,
            MAX_MOVES_PER_BASELINE_BATCH,
        )
        if n_left < MIN_NEW_MOVES_BASELINE:
            logger.info(
                "Caught up: eligible=%s < min=%s (remainder will not train until more data).",
                n_left,
                MIN_NEW_MOVES_BASELINE,
            )
            return 0

        round_i += 1
        logger.info("=== catch-up round %s/%s: train ===", round_i, max_rounds)
        t0 = time.monotonic()
        train = run_baseline_training_pipeline()
        logger.info("Train wall_s=%.1f", time.monotonic() - t0)
        if not _run_ok(train, label="train"):
            return 1

        n_after, cutoff_after = _eligible_count()
        if n_after >= n_left and cutoff_after == cutoff:
            logger.error(
                "Train succeeded but eligible count did not drop "
                "(before=%s after=%s cutoff=%s) — stop to avoid infinite loop.",
                n_left,
                n_after,
                cutoff,
            )
            return 2

        if promote:
            logger.info("=== catch-up round %s/%s: promote ===", round_i, max_rounds)
            t1 = time.monotonic()
            promo = run_baseline_promotion_pipeline()
            logger.info("Promote wall_s=%.1f", time.monotonic() - t1)
            if not _run_ok(promo, label="promote"):
                return 1

    logger.error("Hit max_rounds=%s with eligible still above min — stopping.", max_rounds)
    return 3
