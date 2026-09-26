"""Parent-agnostic contract for a train / score / promote pipeline.

Steps in ``scheme_steps`` call this protocol. They do not know whether the
weights belong to the platform model or one account's personal model.
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


@dataclass(frozen=True)
class PromotionDecision:
    should_promote: bool
    reason: str


class TrainingScheme(Protocol):
    """Data, trainer, and table writes for one model chain."""

    pipeline_name: str
    split_version: str
    min_new_moves: int

    def note_checked(self, db_client: DatabaseClient) -> None:
        """Record that the pending-data gate ran."""

    def count_pending(self, db_client: DatabaseClient) -> int:
        """Unprocessed registry-train moves for this chain."""

    def resolve_parent(self, db_client: DatabaseClient) -> ModelHandle | None:
        """Warm-start weights. None means a cold start."""

    def resolve_reference(self, db_client: DatabaseClient) -> ModelHandle | None:
        """Model currently served for this chain. None means nothing to replace."""

    def load_train_batch(self, db_client: DatabaseClient) -> tuple[list[TrainingDatum], list[str]]:
        """Next registry-train batch and the game ids to mark after a successful fit."""

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
        """Advance the queue for games that were actually fit."""

    def decide_promotion(
        self,
        *,
        candidate_eval: EvalMetrics | None,
        reference: ModelHandle | None,
        reference_eval: EvalMetrics | None,
    ) -> PromotionDecision:
        """Whether the new candidate should replace the served model."""

    def apply_promotion(
        self,
        db_client: DatabaseClient,
        *,
        candidate: ModelHandle,
        reference: ModelHandle | None,
        eval_metrics_json: str | None,
    ) -> ModelHandle:
        """Archive the served row (if any) and mark the candidate as served."""
