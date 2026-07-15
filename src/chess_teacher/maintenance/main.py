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
from chess_teacher.utils.pipeline_utils.pipeline_base import Pipeline
from chess_teacher.utils.pipeline_utils.pipeline_helpers import PipelineRunResult


def run_maintenance() -> PipelineRunResult:
    """Main entry point for maintenance pipeline."""
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
