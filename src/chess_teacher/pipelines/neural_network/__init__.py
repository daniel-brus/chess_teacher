"""Neural-network pipelines (baseline train/promote; user finetune later)."""

from chess_teacher.pipelines.neural_network.main import (
    run_baseline_promotion_pipeline,
    run_baseline_training_pipeline,
)
from chess_teacher.pipelines.neural_network.promotion import PromotionStrategies

__all__ = [
    "PromotionStrategies",
    "run_baseline_promotion_pipeline",
    "run_baseline_training_pipeline",
]
