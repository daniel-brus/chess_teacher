"""Baseline candidate training pipeline entrypoints.

Follow-ups (not wired here yet):
- ``run_baseline_promotion_pipeline`` — evaluate candidate vs production, archive/promote.
- ``run_user_finetune_pipeline(user_id)`` — hook from daily ``PipelineRunner`` after preprocess.
- Held-out validation set with periodic rotation.
"""

from __future__ import annotations

from chess_teacher.pipelines.neural_network.pipeline_steps import (
    CheckSufficientNewDataStep,
    LoadNewDataStep,
    LoadPreviousCandidateWeightsStep,
    LogToMLflowStep,
    TrainIncrementalStep,
    UpdateTrainingStateStep,
)
from chess_teacher.utils.pipeline_utils.pipeline_base import Pipeline
from chess_teacher.utils.pipeline_utils.pipeline_helpers import PipelineRunResult, ProgressWindow


def run_baseline_training_pipeline(
    *,
    progress_window: ProgressWindow | None = None,
) -> PipelineRunResult:
    """Platform-wide incremental baseline train (candidate chain)."""
    pipeline = Pipeline(
        name="baseline_training",
        user_id=None,
        account_id=None,
        steps=[
            CheckSufficientNewDataStep(),
            LoadPreviousCandidateWeightsStep(),
            LoadNewDataStep(),
            TrainIncrementalStep(),
            LogToMLflowStep(),
            UpdateTrainingStateStep(),
        ],
        progress_window=progress_window,
        # Training can exceed the default 1h lock window.
        lock_timeout_hours=6.0,
    )
    return pipeline.run()
