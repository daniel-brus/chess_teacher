from datetime import timedelta

from chess_teacher.pipelines.pipeline_base import (
    PipelineContext,
    PipelineRunResult,
    PipelineStep,
)
from chess_teacher.utils.db_client import DatabaseClient
from chess_teacher.utils.general_utils import get_current_datetime
from chess_teacher.utils.metadata_utils import TableMetadata


class DeleteOldRecordsStep(PipelineStep):
    def __init__(
        self,
        name: str,
        metadata: TableMetadata,
        column: str,
        *,
        retention_period: timedelta,
        additional_where: str | None = None,
    ):
        super().__init__(name, critical=False)
        self.metadata = metadata
        self.column = column
        self.retention_period = retention_period
        self.additional_where = additional_where

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        """Delete old records from the table"""
        cutoff_datetime = get_current_datetime() - self.retention_period
        where = f"""{self.column} < '{cutoff_datetime.isoformat()}'""" + (
            f" AND ({self.additional_where})" if self.additional_where else ""
        )
        rows_deleted = db_client.delete_where(self.metadata, where=where)
        context.progress_update(
            f"Deleted {rows_deleted} old record{'s' if rows_deleted != 1 else ''}."
        )
        self.logger.info(
            f"Deleted {rows_deleted} old records from {self.metadata.qualified_name_sql()} ({where})"
        )
        return


class ClearOrphanedPipelineRunLocksStep(DeleteOldRecordsStep):
    """Clear orphaned pipeline run locks"""

    _RETENTION_PERIOD: timedelta = timedelta(days=1)

    def __init__(self):
        super().__init__(
            "clear_orphaned_pipeline_run_locks",
            PipelineRunResult.get_metadata(),
            "started_at",
            retention_period=self._MAX_TIME_TO_KEEP_LOCK,
            additional_where="finished_at = '1970-01-01 00:00:00+00'",
        )
