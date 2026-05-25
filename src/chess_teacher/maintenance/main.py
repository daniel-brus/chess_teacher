from datetime import timedelta

from chess_teacher.pipelines.pipeline_base import (
    Pipeline,
    PipelineContext,
    PipelineRunResult,
    PipelineStep,
)
from chess_teacher.utils.db_client import DatabaseClient
from chess_teacher.utils.general_utils import get_current_datetime

MAX_TIME_TO_KEEP_LOCK: timedelta = timedelta(days=1)


class ClearPipelineRunLocks(PipelineStep):
    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        """Clear orphaned pipeline run locks"""
        cutoff_datetime = get_current_datetime() - MAX_TIME_TO_KEEP_LOCK
        where = f"""finished_at = '1970-01-01 00:00:00+00' AND "started_at" < '{cutoff_datetime.isoformat()}'"""
        rows_deleted = db_client.delete_where(PipelineRunResult.get_metadata(), where=where)
        self.logger.info(f"Cleared {rows_deleted} orphaned pipeline run locks")


def run_maintenance() -> None:
    """Main entry point for maintenance pipeline"""
    Pipeline(
        name="maintenance",
        steps=[ClearPipelineRunLocks("clear_pipeline_run_locks")],
    ).run()
