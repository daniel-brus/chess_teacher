"""Shared helpers for offline baseline train/eval/promotion (Phases 1-3).

Scripts and ``training_develop.ipynb`` should call these — do not duplicate
split loading or URI scoring in notebook cells.
"""

from __future__ import annotations

from chess_teacher.pipelines.neural_network.create_training_set import (
    TrainingDataStore,
    TrainingDatum,
    account_id_in_sql,
)
from chess_teacher.pipelines.neural_network.models import BaselineModel, BaselineModelStatus
from chess_teacher.pipelines.neural_network.split_registry import get_split_registry
from chess_teacher.pipelines.neural_network.splits import (
    DEFAULT_SPLIT_SALT,
    GameSplitResult,
    SplitBucket,
    game_split_result,
)
from chess_teacher.utils.db.client import DatabaseClient, get_db_client


def account_registry_extra_where(account_id: str) -> str:
    """SQL fragment restricting registry bucket fetches to one ``account_id``."""
    aid = (account_id or "").strip()
    if not aid:
        raise ValueError("account_id is required")
    return account_id_in_sql([aid])


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


def load_registry_bucket_datums(
    db_client: DatabaseClient | None = None,
    *,
    bucket: SplitBucket,
    split_version: str = DEFAULT_SPLIT_SALT,
    limit: int | None = None,
    extra_where: str | None = None,
) -> list[TrainingDatum]:
    """Eligible moves for one registry bucket, lowest ``game_id`` first.

    ``limit`` is a complete-game move cap. ``None`` loads the whole bucket.
    """
    db = db_client or get_db_client()
    return TrainingDataStore(db).fetch_registry_bucket_batch(
        split_version=split_version,
        bucket=bucket.value,
        limit=limit,
        extra_where=extra_where,
    )


def load_registry_val_datums(
    db_client: DatabaseClient | None = None,
    *,
    split_version: str = DEFAULT_SPLIT_SALT,
    limit: int | None = None,
    full: bool = False,
    extra_where: str | None = None,
    assign_if_missing: bool = True,
) -> list[TrainingDatum]:
    """Load registry val: all games, or a lowest-``game_id`` complete-game prefix.

    ``assign_if_missing`` is accepted for call-site compatibility with the
    cutoff/registry sample path; bucket-prefix loads do not assign new games.
    """
    del assign_if_missing  # API compat; unused for bucket-prefix fetch
    if full:
        limit = None
    elif limit is None:
        raise ValueError("limit is required unless full=True")
    return load_registry_bucket_datums(
        db_client,
        bucket=SplitBucket.VAL,
        split_version=split_version,
        limit=limit,
        extra_where=extra_where,
    )


def load_registry_prefix_split(
    db_client: DatabaseClient | None = None,
    *,
    limit: int,
    split_version: str = DEFAULT_SPLIT_SALT,
    include_test: bool = False,
) -> GameSplitResult:
    """Train/val (and optional test) prefixes from the registry, per bucket.

    ``limit`` caps **each** bucket independently (complete games, ``game_id``
    ASC). This is not a mixed timestamp sample.
    """
    train = load_registry_bucket_datums(
        db_client,
        bucket=SplitBucket.TRAIN,
        split_version=split_version,
        limit=limit,
    )
    val = load_registry_bucket_datums(
        db_client,
        bucket=SplitBucket.VAL,
        split_version=split_version,
        limit=limit,
    )
    test: list[TrainingDatum] = []
    if include_test:
        test = load_registry_bucket_datums(
            db_client,
            bucket=SplitBucket.TEST,
            split_version=split_version,
            limit=limit,
        )
    return game_split_result(train, val, test, salt=split_version)


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


def load_account_registry_bucket_datums(
    account_id: str,
    db_client: DatabaseClient | None = None,
    *,
    bucket: SplitBucket,
    split_version: str = DEFAULT_SPLIT_SALT,
    limit: int | None = None,
) -> list[TrainingDatum]:
    """Eligible moves for one account ∩ one registry bucket (``game_id`` ASC).

    Uses the platform hash registry only — never time-orders train/val.
    ``limit`` is a complete-game move cap; ``None`` loads the whole intersection.
    """
    return load_registry_bucket_datums(
        db_client,
        bucket=bucket,
        split_version=split_version,
        limit=limit,
        extra_where=account_registry_extra_where(account_id),
    )


def load_account_registry_split(
    account_id: str,
    db_client: DatabaseClient | None = None,
    *,
    split_version: str = DEFAULT_SPLIT_SALT,
    train_limit: int | None = None,
    val_limit: int | None = None,
    include_test: bool = False,
    test_limit: int | None = None,
) -> GameSplitResult:
    """Train/val/(optional test) for one account via the shared hash registry.

    Each bucket is loaded independently (complete games). Empty buckets are OK
    when that account has no games in that registry bucket yet.
    """
    train = load_account_registry_bucket_datums(
        account_id,
        db_client,
        bucket=SplitBucket.TRAIN,
        split_version=split_version,
        limit=train_limit,
    )
    val = load_account_registry_bucket_datums(
        account_id,
        db_client,
        bucket=SplitBucket.VAL,
        split_version=split_version,
        limit=val_limit,
    )
    test: list[TrainingDatum] = []
    if include_test:
        test = load_account_registry_bucket_datums(
            account_id,
            db_client,
            bucket=SplitBucket.TEST,
            split_version=split_version,
            limit=test_limit,
        )
    return game_split_result(train, val, test, salt=split_version)
