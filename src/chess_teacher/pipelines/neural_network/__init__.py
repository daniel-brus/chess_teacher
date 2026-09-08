"""Neural-network pipelines (baseline train/promote; game-split assignment).

POC package — disposable until Phase 4 greenfield lands
=======================================================
Almost every class/module under ``chess_teacher.pipelines.neural_network`` is
**proof-of-concept**. Keras artifacts in ``ml.baseline_models``, flat-state
``BaselineTrainer``, promotion/random-eval paths, and the new hybrid encoder
are all replaceable. Prefer deleting / rewriting over preserving APIs.

Keep (data, not NN code): preprocessing move characteristics + candidate SF
evals in Postgres. Those feed whatever greenfield trainer wins Phase 2c+.

See ``.agents/docs/ml-training-roadmap.md`` (greenfield / POC assumptions) and
``.agents/docs/ml-phase2c-board-encoder.md``.
"""

from chess_teacher.pipelines.neural_network.main import (
    run_assign_game_splits_pipeline,
    run_baseline_promotion_pipeline,
    run_baseline_training_pipeline,
)
from chess_teacher.pipelines.neural_network.promotion import PromotionStrategies

__all__ = [
    "PromotionStrategies",
    "run_assign_game_splits_pipeline",
    "run_baseline_promotion_pipeline",
    "run_baseline_training_pipeline",
]
