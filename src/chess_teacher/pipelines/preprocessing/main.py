from __future__ import annotations

from chess_teacher.pipelines.modes import PipelineMode
from chess_teacher.pipelines.preprocessing.pipeline_steps import (
    ExtractUserMovesStep,
    RawGamesToGamesStep,
)
from chess_teacher.platform.account import Account
from chess_teacher.utils.pipeline_utils.pipeline_base import Pipeline
from chess_teacher.utils.pipeline_utils.pipeline_helpers import PipelineRunResult, ProgressWindow


def run_preprocessing_pipeline(
    user_id: str,
    account: Account,
    *,
    mode: PipelineMode = PipelineMode.INCREMENTAL,
    progress_window: ProgressWindow | None = None,
) -> PipelineRunResult:
    """Build an account-scoped preprocessing pipeline and run it."""
    pipeline = Pipeline(
        name="preprocessing",
        user_id=user_id,
        account_id=account.account_id,
        steps=[
            RawGamesToGamesStep(mode=mode),
            ExtractUserMovesStep(mode=mode),
        ],
        progress_window=progress_window,
    )
    return pipeline.run()
