"""Shared helpers for offline baseline train/eval/promotion (Phases 1–3).

Scripts and ``training_develop.ipynb`` should call these — do not duplicate
split loading or URI scoring in notebook cells.
"""

from __future__ import annotations

from chess_teacher.pipelines.neural_network.create_training_set import (
    TrainingDataStore,
    TrainingDatum,
)
from chess_teacher.pipelines.neural_network.models import BaselineModel, BaselineModelStatus
from chess_teacher.pipelines.neural_network.split_registry import get_split_registry
from chess_teacher.pipelines.neural_network.splits import (
    DEFAULT_SPLIT_SALT,
    GameSplitResult,
    SplitBucket,
)
from chess_teacher.utils.db.client import DatabaseClient, get_db_client


def load_registry_split(
    db_client: DatabaseClient | None = None,
    *,
    limit: int,
    split_version: str = DEFAULT_SPLIT_SALT,
    assign_if_missing: bool = True,
) -> GameSplitResult:
    """Fetch a cutoff-free sample and partition it via the persistent registry."""
    db = db_client or get_db_client()
    datums, _cutoff = TrainingDataStore(db).fetch_since(None, limit=limit)
    registry = get_split_registry(db, split_version=split_version)
    return registry.split_datums(datums, assign_if_missing=assign_if_missing)


def load_registry_val_datums(
    db_client: DatabaseClient | None = None,
    *,
    split_version: str = DEFAULT_SPLIT_SALT,
    limit: int | None = None,
    full: bool = False,
    assign_if_missing: bool = True,
) -> list[TrainingDatum]:
    """Load registry val moves: either a ``--limit`` sample or all val games."""
    db = db_client or get_db_client()
    if full:
        registry = get_split_registry(db, split_version=split_version)
        game_ids = registry.fetch_game_ids_for_bucket(SplitBucket.VAL)
        return TrainingDataStore(db).fetch_for_game_ids(game_ids)
    if limit is None:
        raise ValueError("limit is required unless full=True")
    split = load_registry_split(
        db,
        limit=limit,
        split_version=split_version,
        assign_if_missing=assign_if_missing,
    )
    return split.val_datums


def resolve_production_model_uri(db_client: DatabaseClient | None = None) -> str:
    """Return the current production baseline artifact URI, or raise."""
    db = db_client or get_db_client()
    row = BaselineModel.latest_with_status(db, BaselineModelStatus.PRODUCTION)
    if row is None or not row.model_uri:
        raise RuntimeError(
            "No production baseline model_uri in ml.baseline_models; "
            "pass --baseline-uri explicitly."
        )
    return row.model_uri
