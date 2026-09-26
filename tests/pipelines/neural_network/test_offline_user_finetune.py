"""Unit tests for Phase 3a account x registry loaders + min-data gate."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chess_teacher.pipelines.neural_network.offline_eval import (
    account_registry_extra_where,
    load_account_registry_bucket_datums,
    load_account_registry_split,
    load_balanced_account_registry_bucket_datums,
)
from chess_teacher.pipelines.neural_network.offline_user_finetune import (
    MIN_USER_TRAIN_MOVES,
    run_offline_user_finetune_eval,
)
from chess_teacher.pipelines.neural_network.splits import SplitBucket


def test_account_registry_extra_where_requires_id() -> None:
    with pytest.raises(ValueError, match="account_id"):
        account_registry_extra_where("")


def test_account_registry_extra_where_is_account_in() -> None:
    clause = account_registry_extra_where("acct-1")
    assert "g.account_id IN" in clause
    assert "acct-1" in clause
    assert "end_time" not in clause


def test_load_account_registry_bucket_forwards_extra_where() -> None:
    store = MagicMock()
    store.fetch_registry_bucket_batch.return_value = []
    db = MagicMock()
    with patch(
        "chess_teacher.pipelines.neural_network.offline_eval.TrainingDataStore",
        return_value=store,
    ):
        load_account_registry_bucket_datums(
            "acct-9",
            db,
            bucket=SplitBucket.TRAIN,
            split_version="baseline-v1",
            limit=100,
        )
    kwargs = store.fetch_registry_bucket_batch.call_args.kwargs
    assert kwargs["bucket"] == "train"
    assert kwargs["split_version"] == "baseline-v1"
    assert kwargs["limit"] == 100
    assert "g.account_id IN" in kwargs["extra_where"]
    assert "acct-9" in kwargs["extra_where"]


def test_load_balanced_account_registry_needs_two_ids() -> None:
    with pytest.raises(ValueError, match=">=2"):
        load_balanced_account_registry_bucket_datums(
            ["only-one"],
            MagicMock(),
            bucket=SplitBucket.TRAIN,
            per_account_limit=100,
        )


def test_load_balanced_account_registry_equal_per_account() -> None:
    calls: list[str] = []

    def _fake_load(account_id: str, *_a: object, **_k: object) -> list[object]:
        calls.append(account_id)
        return [MagicMock(account_id=account_id), MagicMock(account_id=account_id)]

    with patch(
        "chess_teacher.pipelines.neural_network.offline_eval.load_account_registry_bucket_datums",
        side_effect=_fake_load,
    ):
        out = load_balanced_account_registry_bucket_datums(
            ["a1", "a2"],
            MagicMock(),
            bucket=SplitBucket.TRAIN,
            split_version="baseline-v1",
            per_account_limit=25000,
        )
    assert calls == ["a1", "a2"]
    assert len(out) == 4


def test_load_account_registry_split_loads_train_and_val() -> None:
    fake = MagicMock()
    fake.train = (MagicMock(),)
    fake.val = (MagicMock(),)
    fake.salt = "baseline-v1"
    with (
        patch(
            "chess_teacher.pipelines.neural_network.offline_eval.load_account_registry_bucket_datums",
            return_value=[],
        ) as load_bucket,
        patch(
            "chess_teacher.pipelines.neural_network.offline_eval.game_split_result",
            return_value=fake,
        ) as build,
    ):
        split = load_account_registry_split("acct-1", MagicMock(), split_version="baseline-v1")
    assert split is fake
    buckets = [c.kwargs["bucket"] for c in load_bucket.call_args_list]
    assert buckets == [SplitBucket.TRAIN, SplitBucket.VAL]
    assert build.call_args.kwargs["salt"] == "baseline-v1"


def test_user_finetune_skips_below_min_train_moves(tmp_path: Path) -> None:
    parent = tmp_path / "parent.keras"
    parent.write_bytes(b"fake")
    tiny_split = MagicMock()
    tiny_split.train_datums = [MagicMock()] * 10
    tiny_split.val_datums = [MagicMock()] * 20
    tiny_split.salt = "baseline-v1"
    tiny_split.counts = ()
    with (
        patch(
            "chess_teacher.pipelines.neural_network.offline_user_finetune.get_db_client",
            return_value=MagicMock(),
        ),
        patch(
            "chess_teacher.pipelines.neural_network.offline_user_finetune.load_account_registry_split",
            return_value=tiny_split,
        ),
        patch(
            "chess_teacher.pipelines.neural_network.offline_user_finetune.format_split_summary",
            return_value="summary",
        ),
    ):
        code = run_offline_user_finetune_eval(
            account_id="acct-1",
            split_version="baseline-v1",
            parent_weights=str(parent),
            train_parent=False,
            parent_out=None,
            parent_train_limit=None,
            parent_epochs=1,
            parent_style_disagree_boost=2.0,
            train_limit=None,
            val_limit=None,
            epochs=1,
            style_disagree_boost=4.0,
            style_disagree_scale=2.0,
            recency_lambda=None,
            child_out=None,
            min_train_moves=MIN_USER_TRAIN_MOVES,
        )
    assert code == 2


def test_user_finetune_requires_parent_source() -> None:
    code = run_offline_user_finetune_eval(
        account_id="acct-1",
        split_version="baseline-v1",
        parent_weights=None,
        train_parent=False,
        parent_out=None,
        parent_train_limit=None,
        parent_epochs=1,
        parent_style_disagree_boost=2.0,
        train_limit=None,
        val_limit=None,
        epochs=1,
        style_disagree_boost=4.0,
        style_disagree_scale=2.0,
        recency_lambda=None,
        child_out=None,
    )
    assert code == 1
