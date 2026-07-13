from chess_teacher.maintenance.pipeline_steps import (
    AggregateExceptionHourlyStep,
    AggregateLogLevelHourlyStep,
    ClearOrphanedPipelineRunLocksStep,
    DeleteOldRawLogsStep,
    DeleteOldS3LogFilesStep,
    DeleteOldWarningErrorLogsStep,
    LoadRawLogsStep,
    PromoteWarningErrorLogsStep,
)
from chess_teacher.utils.pipeline_utils.pipeline_base import Pipeline


def run_maintenance() -> None:
    """Main entry point for maintenance pipeline"""
    Pipeline(
        name="maintenance",
        steps=[
            LoadRawLogsStep(),
            PromoteWarningErrorLogsStep(),
            AggregateLogLevelHourlyStep(),
            AggregateExceptionHourlyStep(),
            DeleteOldRawLogsStep(),
            DeleteOldWarningErrorLogsStep(),
            DeleteOldS3LogFilesStep(),
            ClearOrphanedPipelineRunLocksStep(),
        ],
    ).run()
