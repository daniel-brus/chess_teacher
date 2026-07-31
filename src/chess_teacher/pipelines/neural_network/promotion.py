"""Swappable promotion strategies: eval set, scoring, decide.

Default scorer: user-move top-k accuracy (style imitation). ActionMaeScorer kept for legacy MSE.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from chess_teacher.pipelines.neural_network.create_training_set import (
    TrainingBatch,
    TrainingDataStore,
    TrainingDatum,
)
from chess_teacher.pipelines.neural_network.mlflow_utils import MLflowTracker
from chess_teacher.pipelines.neural_network.move_encoding import POLICY_VOCAB_SIZE
from chess_teacher.utils.db.client import DatabaseClient
from chess_teacher.utils.logging import get_logger

logger = get_logger()

# Thin-promotion defaults (not env — swap via constructor / subclasses later).
DEFAULT_EVAL_SAMPLE_SIZE = 2_000
DEFAULT_EVAL_RANDOM_SEED = 42
DEFAULT_PROMOTION_MARGIN = 0.0
DEFAULT_TOP_K = 1


@dataclass(frozen=True)
class ModelScore:
    """Comparable score for promotion. ``primary`` is the decision value."""

    primary: float
    higher_is_better: bool
    details: dict[str, float] = field(default_factory=dict)

    def beats(self, other: ModelScore, *, margin: float = 0.0) -> bool:
        """True if this score is at least as good as ``other`` within ``margin``."""
        if self.higher_is_better != other.higher_is_better:
            raise ValueError("Cannot compare ModelScore with opposite higher_is_better flags")
        if self.higher_is_better:
            return self.primary + margin >= other.primary
        return self.primary - margin <= other.primary


@dataclass(frozen=True)
class PromotionVerdict:
    should_promote: bool
    reason: str
    candidate_score: ModelScore | None = None
    production_score: ModelScore | None = None


class EvalSetProvider(ABC):
    """Provides datums used only for scoring (not for training in this pipeline)."""

    @abstractmethod
    def sample(self, db_client: DatabaseClient) -> list[TrainingDatum]:
        """Return eval datums. Empty list → promotion should skip."""


class RandomEvalSetProvider(EvalSetProvider):
    """Temporary shortcut: random moves with characteristics.

    Replace later with a fixed / rotating held-out set that training excludes.
    """

    def __init__(
        self,
        *,
        size: int = DEFAULT_EVAL_SAMPLE_SIZE,
        seed: int = DEFAULT_EVAL_RANDOM_SEED,
    ) -> None:
        self.size = size
        self.seed = seed

    def sample(self, db_client: DatabaseClient) -> list[TrainingDatum]:
        datums = TrainingDataStore(db_client).fetch_random(limit=self.size, seed=self.seed)
        logger.info(
            "RandomEvalSetProvider sampled n=%s size=%s seed=%s",
            len(datums),
            self.size,
            self.seed,
        )
        return datums


class ModelScorer(ABC):
    """Scores a model artifact URI on an eval set."""

    @abstractmethod
    def score(self, *, model_uri: str, datums: list[TrainingDatum]) -> ModelScore:
        """Return a ModelScore. Must set higher_is_better consistently."""


class ActionMaeScorer(ModelScorer):
    """Legacy stub metric: MAE on ``action_label`` vectors (lower is better)."""

    def __init__(self, *, tracker: MLflowTracker | None = None) -> None:
        self._tracker = tracker or MLflowTracker()

    def score(self, *, model_uri: str, datums: list[TrainingDatum]) -> ModelScore:
        if not datums:
            raise ValueError("ActionMaeScorer.score requires a non-empty eval set")

        from tensorflow import keras  # type: ignore[import-untyped]

        weights_path = self._tracker.require_keras_weights(model_uri)
        model = keras.models.load_model(weights_path)
        batch = TrainingBatch(datums)
        x = batch.state_matrix()
        y_true = batch.action_matrix()
        y_pred = np.asarray(model.predict(x, verbose=0), dtype=np.float32)
        mae = float(np.mean(np.abs(y_pred - y_true)))
        mse = float(np.mean((y_pred - y_true) ** 2))
        return ModelScore(
            primary=mae,
            higher_is_better=False,
            details={"mae": mae, "mse": mse, "n_eval": float(len(datums))},
        )


class TopKMoveAccuracyScorer(ModelScorer):
    """User-move top-k hit rate after illegal-move mask (higher is better).

    Measures style imitation: would the model rank the user's played move in its
    top-k legal predictions — not Stockfish best-move agreement.
    """

    def __init__(
        self,
        *,
        k: int = DEFAULT_TOP_K,
        tracker: MLflowTracker | None = None,
        vocab_size: int = POLICY_VOCAB_SIZE,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self.vocab_size = vocab_size
        self._tracker = tracker or MLflowTracker()

    def score(self, *, model_uri: str, datums: list[TrainingDatum]) -> ModelScore:
        if not datums:
            raise ValueError("TopKMoveAccuracyScorer.score requires a non-empty eval set")

        from chess_teacher.pipelines.neural_network.train import load_policy_from_uri

        model = load_policy_from_uri(
            model_uri,
            tracker=self._tracker,
            vocab_size=self.vocab_size,
        )

        batch = TrainingBatch(datums)
        x = batch.state_matrix()
        y_index, legal_mask = batch.policy_targets()
        logits = np.asarray(model.predict(x, verbose=0), dtype=np.float64)
        if logits.ndim != 2 or logits.shape[1] != self.vocab_size:
            raise ValueError(f"Unexpected logits shape {logits.shape}")

        masked = np.where(legal_mask, logits, -np.inf)
        if self.k == 1:
            top = np.argmax(masked, axis=1)
            hits = top == y_index
        else:
            k_eff = min(self.k, self.vocab_size)
            part = np.argpartition(masked, -k_eff, axis=1)[:, -k_eff:]
            hits = np.any(part == y_index.reshape(-1, 1), axis=1)

        top1 = np.argmax(masked, axis=1) == y_index
        k5 = min(5, self.vocab_size)
        part5 = np.argpartition(masked, -k5, axis=1)[:, -k5:]
        top5 = np.any(part5 == y_index.reshape(-1, 1), axis=1)

        acc = float(np.mean(hits))
        acc1 = float(np.mean(top1))
        acc5 = float(np.mean(top5))
        return ModelScore(
            primary=acc,
            higher_is_better=True,
            details={
                f"top{self.k}_accuracy": acc,
                "top1_accuracy": acc1,
                "top5_accuracy": acc5,
                "n_eval": float(len(datums)),
                "vocab_size": float(self.vocab_size),
            },
        )


class PromotionPolicy(ABC):
    """Decides whether candidate should replace production."""

    @abstractmethod
    def decide(
        self,
        *,
        candidate_score: ModelScore | None,
        production_score: ModelScore | None,
        has_candidate: bool,
        has_production: bool,
    ) -> PromotionVerdict: ...


class BetterOrEqualPromotionPolicy(PromotionPolicy):
    """Promote if no production yet, or candidate is not worse than production (within margin)."""

    def __init__(self, *, margin: float = DEFAULT_PROMOTION_MARGIN) -> None:
        self.margin = margin

    def decide(
        self,
        *,
        candidate_score: ModelScore | None,
        production_score: ModelScore | None,
        has_candidate: bool,
        has_production: bool,
    ) -> PromotionVerdict:
        if not has_candidate:
            return PromotionVerdict(
                should_promote=False,
                reason="No candidate baseline model to promote.",
            )
        if not has_production:
            return PromotionVerdict(
                should_promote=True,
                reason="No production baseline yet; auto-promote candidate.",
                candidate_score=candidate_score,
                production_score=None,
            )
        if candidate_score is None or production_score is None:
            return PromotionVerdict(
                should_promote=False,
                reason="Missing scores for candidate and/or production.",
                candidate_score=candidate_score,
                production_score=production_score,
            )
        if candidate_score.beats(production_score, margin=self.margin):
            direction = "higher" if candidate_score.higher_is_better else "lower"
            return PromotionVerdict(
                should_promote=True,
                reason=(
                    f"Candidate primary={candidate_score.primary:.6f} "
                    f"{direction}-is-better vs production={production_score.primary:.6f} "
                    f"(margin={self.margin})."
                ),
                candidate_score=candidate_score,
                production_score=production_score,
            )
        return PromotionVerdict(
            should_promote=False,
            reason=(
                f"Candidate primary={candidate_score.primary:.6f} did not beat "
                f"production={production_score.primary:.6f} (margin={self.margin})."
            ),
            candidate_score=candidate_score,
            production_score=production_score,
        )


@dataclass(frozen=True)
class PromotionStrategies:
    """Bundle of interchangeable promotion collaborators."""

    eval_set: EvalSetProvider = field(default_factory=RandomEvalSetProvider)
    scorer: ModelScorer = field(default_factory=TopKMoveAccuracyScorer)
    policy: PromotionPolicy = field(default_factory=BetterOrEqualPromotionPolicy)
