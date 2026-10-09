"""Contract for one training run.

``user_id`` none is the platform model. A user id pools that user's linked
accounts. Steps do not branch on which one it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from chess_teacher.pipelines.neural_network.create_training_set import TrainingDatum
from chess_teacher.pipelines.neural_network.eval_metrics import EvalMetrics
from chess_teacher.utils.db.client import DatabaseClient


@dataclass(frozen=True)
class ModelHandle:
    """One stored model the steps can warm-start from or score against.

    ``kind`` is for the scheme that writes the next row. Steps pass the handle
    through without branching on it.
    """

    key: str
    weights_uri: str | None
    compatible: bool
    kind: str
    payload: Any = None
    parent_baseline_version: str | None = None


@dataclass(frozen=True)
class PromotionDecision:
    should_promote: bool
    reason: str


@dataclass(frozen=True)
class ParentBaselineDecision:
    """Whether a promoted platform model becomes the parent baseline.

    Separate from promotion. A model can be served for play without moving
    the pointer that later personal runs warm-start from.
    """

    should_adopt: bool
    reason: str


class TrainingScheme(Protocol):
    """Data, trainer, and table writes for one model chain."""

    pipeline_name: str
    split_version: str
    min_new_moves: int
    considers_parent_baseline: bool

    def note_checked(self, db_client: DatabaseClient) -> None:
        """Record that the pending-data gate ran."""

    def align_personal_queue(self, db_client: DatabaseClient) -> None:
        """Replay a personal queue once when its parent baseline changes."""

    def count_pending(self, db_client: DatabaseClient) -> int:
        """Registry-train moves this round may train."""

    def resolve_parent(self, db_client: DatabaseClient) -> ModelHandle | None:
        """Warm-start and comparison model. None means a cold start."""

    def load_train_batch(self, db_client: DatabaseClient) -> tuple[list[TrainingDatum], list[str]]:
        """Next registry-train batch. Never-tried games come before one-miss games."""

    def load_eval_datums(self, db_client: DatabaseClient) -> list[TrainingDatum]:
        """Frozen registry-val moves. Same list is reused for every score in the run."""

    def fit(
        self,
        datums: list[TrainingDatum],
        *,
        weights_path: Path | None,
    ) -> tuple[Any, dict[str, float]]:
        """Train. Called only from the TensorFlow step."""

    def save(self, model: Any, path: Path) -> None:
        """Write ``model`` to ``path``."""

    def next_version(self, db_client: DatabaseClient) -> str:
        """Version label for the candidate about to be recorded."""

    def insert_candidate(
        self,
        db_client: DatabaseClient,
        *,
        version: str,
        model_uri: str,
        mlflow_run_id: str | None,
        metrics_json: str,
        parent: ModelHandle | None,
    ) -> ModelHandle:
        """Insert the new candidate row and return a handle to it."""

    def mark_trained(self, db_client: DatabaseClient, game_ids: list[str]) -> None:
        """Take promoted games out of the queue. They are in the served model."""

    def note_rejected(self, db_client: DatabaseClient, game_ids: list[str]) -> None:
        """Move a batch that missed promotion behind never-tried games."""

    def decide_promotion(
        self,
        *,
        candidate_eval: EvalMetrics | None,
        parent: ModelHandle | None,
        parent_eval: EvalMetrics | None,
    ) -> PromotionDecision:
        """Whether the new candidate should replace the parent when it is the same chain."""

    def apply_promotion(
        self,
        db_client: DatabaseClient,
        *,
        candidate: ModelHandle,
        current: ModelHandle | None,
        eval_metrics_json: str | None,
    ) -> ModelHandle:
        """Archive ``current`` (if any) and mark the candidate as served."""

    def current_parent_baseline(self, db_client: DatabaseClient) -> ModelHandle | None:
        """Platform model personal runs warm-start from. None if the pointer is unset."""

    def decide_parent_baseline(
        self,
        *,
        candidate_eval: EvalMetrics | None,
        parent: ModelHandle | None,
        parent_eval: EvalMetrics | None,
    ) -> ParentBaselineDecision:
        """Whether a promoted platform candidate should replace the parent baseline."""

    def apply_parent_baseline(
        self,
        db_client: DatabaseClient,
        *,
        candidate: ModelHandle,
    ) -> ModelHandle:
        """Point the parent baseline at ``candidate`` without changing who is served."""
