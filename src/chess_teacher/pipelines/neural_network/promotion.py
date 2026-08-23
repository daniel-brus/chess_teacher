"""Swappable promotion strategies: eval set, scoring, decide.

Default scorer: candidate-style top-k accuracy (user move among SF-featured legals).
ActionMaeScorer kept for legacy MSE.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from chess_teacher.pipelines.neural_network.candidate_eval import MAX_CANDIDATES
from chess_teacher.pipelines.neural_network.create_training_set import (
    TrainingBatch,
    TrainingDataStore,
    TrainingDatum,
)
from chess_teacher.pipelines.neural_network.mlflow_utils import MLflowTracker
from chess_teacher.pipelines.neural_network.ply_weights import candidate_style_sample_weights
from chess_teacher.pipelines.neural_network.tf_runtime import ensure_tensorflow_logging
from chess_teacher.utils.db.client import DatabaseClient
from chess_teacher.utils.logging import get_logger

logger = get_logger()
ensure_tensorflow_logging()

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
    """Temporary shortcut: random moves with candidate_evaluations.

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

        ensure_tensorflow_logging()
        from tensorflow import keras  # type: ignore[import-untyped]

        ensure_tensorflow_logging()
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


class CandidateStyleTopKScorer(ModelScorer):
    """User-move top-k among SF-featured candidates (higher is better).

    Prefer precomputed ``candidate_evaluations`` on eval datums (fast). Uses the
    same sample weights as training (ply * style disagree boost) for primary.
    """

    def __init__(
        self,
        *,
        k: int = DEFAULT_TOP_K,
        tracker: MLflowTracker | None = None,
        max_candidates: int = MAX_CANDIDATES,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self.max_candidates = max_candidates
        self._tracker = tracker or MLflowTracker()

    def score(self, *, model_uri: str, datums: list[TrainingDatum]) -> ModelScore:
        if not datums:
            raise ValueError("CandidateStyleTopKScorer.score requires a non-empty eval set")

        from chess_teacher.pipelines.neural_network.train import (
            load_candidate_style_from_uri,
        )

        model = load_candidate_style_from_uri(
            model_uri,
            tracker=self._tracker,
            max_candidates=self.max_candidates,
        )

        batch = TrainingBatch(datums)
        feats, mask, labels, kept = batch.candidate_style_targets()
        if not kept:
            raise ValueError(
                "CandidateStyleTopKScorer: no eval datums with usable candidate_evaluations"
            )
        kept_datums = [datums[i] for i in kept]
        x_state = TrainingBatch(kept_datums).state_matrix()
        logits = np.asarray(
            model.predict({"state": x_state, "move_feats": feats}, verbose=0),
            dtype=np.float64,
        )
        if logits.ndim != 2 or logits.shape[1] != self.max_candidates:
            raise ValueError(f"Unexpected logits shape {logits.shape}")

        masked = np.where(mask > 0.5, logits, -np.inf)
        y_index = labels
        if self.k == 1:
            hits = np.argmax(masked, axis=1) == y_index
        else:
            k_eff = min(self.k, self.max_candidates)
            part = np.argpartition(masked, -k_eff, axis=1)[:, -k_eff:]
            hits = np.any(part == y_index.reshape(-1, 1), axis=1)

        top1 = np.argmax(masked, axis=1) == y_index
        k3 = min(3, self.max_candidates)
        part3 = np.argpartition(masked, -k3, axis=1)[:, -k3:]
        top3 = np.any(part3 == y_index.reshape(-1, 1), axis=1)

        weights = candidate_style_sample_weights(
            [d.ply for d in kept_datums],
            feats,
            labels,
        )
        w_sum = float(np.sum(weights))
        acc_w = float(np.sum(hits.astype(np.float64) * weights) / w_sum) if w_sum else 0.0
        acc1 = float(np.mean(top1))
        acc3 = float(np.mean(top3))
        acc = float(np.mean(hits))
        return ModelScore(
            primary=acc_w,
            higher_is_better=True,
            details={
                f"top{self.k}_accuracy": acc,
                f"top{self.k}_accuracy_ply_weighted": acc_w,
                f"top{self.k}_accuracy_weighted": acc_w,
                "top1_accuracy": acc1,
                "top3_accuracy": acc3,
                "n_eval": float(len(kept_datums)),
                "n_dropped": float(len(datums) - len(kept_datums)),
                "max_candidates": float(self.max_candidates),
                "head_candidate_style": 1.0,
            },
        )


# Backward-compatible name used by older imports / PromotionStrategies default.
TopKMoveAccuracyScorer = CandidateStyleTopKScorer


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
    scorer: ModelScorer = field(default_factory=CandidateStyleTopKScorer)
    policy: PromotionPolicy = field(default_factory=BetterOrEqualPromotionPolicy)
