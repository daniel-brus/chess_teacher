from __future__ import annotations

from chess_teacher.pipelines.ingestion.pipeline_steps import (
    ArchiveIngestedFilesStep,
    ExtractUserMovesStep,
    IngestionFromAPIStreamStep,
    LoadIngestedFilesToDB,
)
from chess_teacher.platform.account import Account
from chess_teacher.utils.pipeline_utils.pipeline_base import Pipeline
from chess_teacher.utils.pipeline_utils.pipeline_helpers import PipelineRunResult, ProgressWindow


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
            ExtractUserMovesStep(),
        ],
        progress_window=progress_window,
    )
    return pipeline.run()
