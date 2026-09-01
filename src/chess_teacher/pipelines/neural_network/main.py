"""Baseline training + promotion pipeline entrypoints.

Roadmap: ``.agents/docs/ml-training-roadmap.md`` (Phase 4+ wires held-out eval here).

Follow-ups (not wired here yet):
- ``run_user_finetune_pipeline(user_id)`` — hook from daily ``PipelineRunner`` after preprocess.
- Held-out validation set with periodic rotation (replace ``RandomEvalSetProvider``).
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
from chess_teacher.pipelines.neural_network.promotion import PromotionStrategies
from chess_teacher.pipelines.neural_network.promotion_steps import (
    ApplyPromotionStep,
    DecidePromotionStep,
    LoadPromotionModelsStep,
    SampleEvalSetStep,
    ScoreModelsStep,
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


def run_baseline_promotion_pipeline(
    *,
    strategies: PromotionStrategies | None = None,
    progress_window: ProgressWindow | None = None,
) -> PipelineRunResult:
    """Compare candidate vs production; promote when policy says so.

    Inject custom ``PromotionStrategies`` to swap eval set / scorer / policy later.
    """
    bundle = strategies or PromotionStrategies()
    pipeline = Pipeline(
        name="baseline_promotion",
        user_id=None,
        account_id=None,
        steps=[
            LoadPromotionModelsStep(),
            SampleEvalSetStep(bundle),
            ScoreModelsStep(bundle),
            DecidePromotionStep(bundle),
            ApplyPromotionStep(),
        ],
        progress_window=progress_window,
        lock_timeout_hours=2.0,
    )
    return pipeline.run()
