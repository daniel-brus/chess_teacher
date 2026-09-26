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
from chess_teacher.utils.pipeline_utils.pipeline_base import Pipeline
from chess_teacher.utils.pipeline_utils.pipeline_helpers import PipelineRunResult


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


def run_nightly_maintenance() -> tuple[PipelineRunResult, PipelineRunResult]:
    """Nightly job: maintenance, then platform baseline train and promote."""
    maintenance = run_maintenance()
    training = run_baseline_training_pipeline(promote=True)
    return maintenance, training
