from __future__ import annotations

from chess_teacher.ingestion.pipeline_steps import (
    ArchiveIngestedFilesStep,
    IngestionFromAPIStreamStep,
    LoadIngestedFilesToDB,
)
from chess_teacher.pipelines.pipeline_base import Pipeline
from chess_teacher.pipelines.pipeline_helpers import PipelineRunResult, ProgressWindow
from chess_teacher.platform.account import Account


def run_ingestion_pipeline(
    user_id: str,
    account: Account,
    *,
    progress_window: ProgressWindow | None = None,
) -> PipelineRunResult:
    """Build an account-scoped ingestion pipeline and run it."""
    pipeline = Pipeline(
        name="ingestion",
        user_id=user_id,
        account_id=account.account_id,
        steps=[
            IngestionFromAPIStreamStep(),
            LoadIngestedFilesToDB(),
            ArchiveIngestedFilesStep(),
        ],
        progress_window=progress_window,
    )
    return pipeline.run()
