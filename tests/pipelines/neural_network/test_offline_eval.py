"""Unit tests for offline eval helpers (no Keras / no DB)."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest

from chess_teacher.pipelines.neural_network import offline_eval
from chess_teacher.pipelines.neural_network.offline_eval import (
    load_registry_prefix_split,
    load_registry_val_datums,
)
from chess_teacher.pipelines.neural_network.splits import SplitBucket


def test_offline_eval_has_no_timestamp_sample_loader() -> None:
    src = inspect.getsource(offline_eval)
    assert "load_registry_split" not in src
    assert "fetch_since" not in src


def test_load_registry_val_datums_requires_limit_unless_full() -> None:
    with pytest.raises(ValueError, match="limit is required"):
        load_registry_val_datums(MagicMock(), full=False, limit=None)


def test_load_registry_val_datums_uses_game_id_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MagicMock()
    store.fetch_registry_bucket_batch.return_value = [MagicMock(game_id="v1")]
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.offline_eval.TrainingDataStore",
        lambda db: store,
    )
    db = MagicMock()
    out = load_registry_val_datums(db, split_version="baseline-v1", limit=50, full=False)
    assert len(out) == 1
    store.fetch_registry_bucket_batch.assert_called_once_with(
        split_version="baseline-v1",
        bucket=SplitBucket.VAL.value,
        limit=50,
        extra_where=None,
    )


def test_load_registry_val_datums_full_has_no_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MagicMock()
    store.fetch_registry_bucket_batch.return_value = []
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.offline_eval.TrainingDataStore",
        lambda db: store,
    )
    load_registry_val_datums(MagicMock(), full=True, limit=10)
    assert store.fetch_registry_bucket_batch.call_args.kwargs["limit"] is None


def test_load_registry_prefix_split_caps_each_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MagicMock()
    store.fetch_registry_bucket_batch.side_effect = [
        [MagicMock(game_id="t1") for _ in range(3)],
        [MagicMock(game_id="v1") for _ in range(2)],
        [MagicMock(game_id="x1")],
    ]
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.offline_eval.TrainingDataStore",
        lambda db: store,
    )
    split = load_registry_prefix_split(
        MagicMock(),
        limit=40,
        split_version="baseline-v1",
        include_test=True,
    )
    buckets = [call.kwargs["bucket"] for call in store.fetch_registry_bucket_batch.call_args_list]
    assert buckets == [
        SplitBucket.TRAIN.value,
        SplitBucket.VAL.value,
        SplitBucket.TEST.value,
    ]
    assert all(
        call.kwargs["limit"] == 40 for call in store.fetch_registry_bucket_batch.call_args_list
    )
    assert len(split.train_datums) == 3
    assert len(split.val_datums) == 2
    assert len(split.test_datums) == 1
