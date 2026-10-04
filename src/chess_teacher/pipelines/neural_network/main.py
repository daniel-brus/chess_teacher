"""Baseline training + promotion pipeline entrypoints.

Roadmap: ``.agents/docs/ml-training-roadmap.md``.

``run_baseline_training_pipeline`` and ``run_personal_training_pipeline`` share
the steps in ``scheme_steps``. ``user_id`` none is the platform model. A user
id pools that user's accounts. Promotion entrypoint below is still the legacy
random-eval chain.
"""

from __future__ import annotations

from chess_teacher.pipelines.neural_network.promotion import PromotionStrategies
from chess_teacher.pipelines.neural_network.promotion_steps import (
    ApplyPromotionStep,
    DecidePromotionStep,
    LoadPromotionModelsStep,
    SampleEvalSetStep,
    ScoreModelsStep,
)
from chess_teacher.pipelines.neural_network.scheme_steps import build_training_scheme_steps
from chess_teacher.pipelines.neural_network.schemes import ModelTraining
from chess_teacher.pipelines.neural_network.training_scheme import TrainingScheme
from chess_teacher.utils.pipeline_utils.pipeline_base import Pipeline
from chess_teacher.utils.pipeline_utils.pipeline_helpers import PipelineRunResult, ProgressWindow


def run_training_scheme_pipeline(
    scheme: TrainingScheme,
    *,
    promote: bool = False,
    user_id: str | None = None,
    account_id: str | None = None,
    progress_window: ProgressWindow | None = None,
) -> PipelineRunResult:
    """Train, score on the scheme's registry val, and optionally promote."""
    pipeline = Pipeline(
        name=scheme.pipeline_name,
        user_id=user_id,
        account_id=account_id,
        steps=build_training_scheme_steps(scheme, promote=promote),
        progress_window=progress_window,
        # Training can exceed the default 1h lock window.
        lock_timeout_hours=6.0,
    )
    return pipeline.run()


def run_baseline_training_pipeline(
    *,
    promote: bool = False,
    progress_window: ProgressWindow | None = None,
) -> PipelineRunResult:
    """Platform-wide incremental baseline train (registry-train queue)."""
    return run_training_scheme_pipeline(
        ModelTraining(),
        promote=promote,
        progress_window=progress_window,
    )


def run_personal_training_pipeline(
    user_id: str,
    *,
    promote: bool = False,
    progress_window: ProgressWindow | None = None,
) -> PipelineRunResult:
    """Finetune one user on every linked account's registry-train moves."""
    return run_training_scheme_pipeline(
        ModelTraining(user_id),
        promote=promote,
        user_id=user_id,
        progress_window=progress_window,
    )


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
