from chess_teacher.maintenance.pipeline_steps import (
    AggregateExceptionHourlyStep,
    AggregateLogLevelHourlyStep,
    ClearOrphanedPipelineRunLocksStep,
    DeleteOldRawLogsStep,
    DeleteOldS3LogFilesStep,
    DeleteOldWarningErrorLogsStep,
    InvalidateAdminLogDashboardCacheStep,
    LoadRawLogsStep,
    PromoteWarningErrorLogsStep,
)
from chess_teacher.pipelines.neural_network.main import run_baseline_training_pipeline
from chess_teacher.pipelines.neural_network.schemes import ModelTraining
from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.env_utils import get_optional_env_variable
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.pipeline_utils.pipeline_base import Pipeline
from chess_teacher.utils.pipeline_utils.pipeline_helpers import (
    PipelineResult,
    PipelineRunResult,
)

logger = get_logger()

BASELINE_NIGHTLY_ROUNDS_ENV = "BASELINE_NIGHTLY_ROUNDS"


def resolve_baseline_nightly_rounds(raw: str | None = None) -> int:
    """How many baseline train-and-promote rounds the nightly job may run.

    Empty or invalid values stay at 1. A later round runs only when enough
    unprocessed registry-train moves remain after the previous round.
    """
    text = raw if raw is not None else get_optional_env_variable(BASELINE_NIGHTLY_ROUNDS_ENV)
    text = text.strip()
    if not text:
        return 1
    try:
        value = int(text)
    except ValueError:
        logger.warning("%s=%r is not an integer; using 1.", BASELINE_NIGHTLY_ROUNDS_ENV, text)
        return 1
    if value < 1:
        logger.warning("%s=%s is below 1; using 1.", BASELINE_NIGHTLY_ROUNDS_ENV, value)
        return 1
    return value


def run_maintenance() -> PipelineRunResult:
    """Log cleanup and orphan-lock maintenance. Does not train."""
    return Pipeline(
        name="maintenance",
        steps=[
            LoadRawLogsStep(),
            PromoteWarningErrorLogsStep(),
            AggregateLogLevelHourlyStep(),
            AggregateExceptionHourlyStep(),
            InvalidateAdminLogDashboardCacheStep(),
            DeleteOldRawLogsStep(),
            DeleteOldWarningErrorLogsStep(),
            DeleteOldS3LogFilesStep(),
            ClearOrphanedPipelineRunLocksStep(),
        ],
    ).run()


def run_nightly_baseline_rounds(rounds: int) -> tuple[PipelineRunResult, ...]:
    """Run up to ``rounds`` platform train-and-promote passes.

    The first pass always runs, including when it will skip for lack of data.
    Later passes run only while unprocessed registry-train moves stay at or
    above the training minimum. A failed pass stops the loop.
    """
    rounds = max(1, int(rounds))
    scheme = ModelTraining()
    db_client = get_db_client()
    results: list[PipelineRunResult] = []
    for index in range(rounds):
        if index > 0:
            pending = scheme.count_pending(db_client)
            if pending < scheme.min_new_moves:
                logger.info(
                    "Nightly baseline stopping after %s round(s): pending=%s < min=%s",
                    index,
                    pending,
                    scheme.min_new_moves,
                )
                break
        logger.info("Nightly baseline round %s/%s", index + 1, rounds)
        result = run_baseline_training_pipeline(promote=True)
        results.append(result)
        if result.result in {PipelineResult.FAILURE, PipelineResult.PARTIAL}:
            logger.warning(
                "Nightly baseline round %s/%s stopped the loop result=%s",
                index + 1,
                rounds,
                result.result.value,
            )
            break
    return tuple(results)


def run_nightly_maintenance(
    *,
    baseline_rounds: int = 1,
) -> tuple[PipelineRunResult, tuple[PipelineRunResult, ...]]:
    """Nightly job: maintenance, then one or more baseline train-and-promote rounds."""
    maintenance = run_maintenance()
    training = run_nightly_baseline_rounds(baseline_rounds)
    return maintenance, training
