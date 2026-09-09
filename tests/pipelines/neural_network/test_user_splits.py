"""Unit tests for account hash-split (same buckets as game_split_bucket)."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

from chess_teacher.pipelines.neural_network import user_splits
from chess_teacher.pipelines.neural_network.splits import (
    DEFAULT_SPLIT_SALT,
    SplitBucket,
    game_split_bucket,
)
from chess_teacher.pipelines.neural_network.user_splits import (
    cap_train_game_ids_by_game_id,
    load_account_split,
    partition_account_game_ids,
    split_account_datums,
)


def _datum(game_id: str, n: int = 1) -> list[SimpleNamespace]:
    return [SimpleNamespace(game_id=game_id) for _ in range(n)]


def _ids_for_bucket(bucket: SplitBucket, n: int, *, start: int = 0) -> tuple[list[str], int]:
    out: list[str] = []
    i = start
    while len(out) < n:
        gid = f"syn-{i}"
        if game_split_bucket(gid, salt=DEFAULT_SPLIT_SALT) is bucket:
            out.append(gid)
        i += 1
    return out, i


def test_user_splits_uses_game_split_bucket() -> None:
    src = inspect.getsource(user_splits)
    assert "game_split_bucket" in src
    assert "DEFAULT_SPLIT_SALT" in src
    assert "split_registry" not in src
    assert "load_account_holdout" not in src
    assert "AccountHoldout" not in src
    assert "ingested_at" not in src


def test_partition_matches_game_split_bucket() -> None:
    ids = [f"game-{i}" for i in range(80)]
    train_ids, val_ids, test_ids = partition_account_game_ids(ids)
    for gid in train_ids:
        assert game_split_bucket(gid, salt=DEFAULT_SPLIT_SALT) is SplitBucket.TRAIN
    for gid in val_ids:
        assert game_split_bucket(gid, salt=DEFAULT_SPLIT_SALT) is SplitBucket.VAL
    for gid in test_ids:
        assert game_split_bucket(gid, salt=DEFAULT_SPLIT_SALT) is SplitBucket.TEST
    assert set(train_ids + val_ids + test_ids) == set(ids)


def test_hash_buckets_can_be_nonempty() -> None:
    train_ids, val_ids, test_ids = partition_account_game_ids(f"syn-{i}" for i in range(400))
    assert train_ids
    assert val_ids
    assert test_ids


def test_split_account_datums_same_buckets_as_hash() -> None:
    ids = [f"acct-game-{i}" for i in range(40)]
    datums: list[SimpleNamespace] = []
    for gid in ids:
        datums.extend(_datum(gid, n=2))
    split = split_account_datums(datums)  # type: ignore[arg-type]
    assert split.salt == DEFAULT_SPLIT_SALT
    for d in split.train:
        assert game_split_bucket(d.game_id, salt=DEFAULT_SPLIT_SALT) is SplitBucket.TRAIN
    for d in split.val:
        assert game_split_bucket(d.game_id, salt=DEFAULT_SPLIT_SALT) is SplitBucket.VAL
    for d in split.test:
        assert game_split_bucket(d.game_id, salt=DEFAULT_SPLIT_SALT) is SplitBucket.TEST
    assert all(
        sum(1 for d in (*split.train, *split.val, *split.test) if d.game_id == gid) == 2
        for gid in ids
    )


def test_load_account_split_fetches_val_test_then_capped_train() -> None:
    train_ids, i = _ids_for_bucket(SplitBucket.TRAIN, 1)
    val_ids, i = _ids_for_bucket(SplitBucket.VAL, 1, start=i)
    test_ids, _i = _ids_for_bucket(SplitBucket.TEST, 1, start=i)
    t0 = datetime(2026, 5, 1, tzinfo=UTC)
    t1 = datetime(2026, 5, 2, tzinfo=UTC)
    t2 = datetime(2026, 5, 3, tzinfo=UTC)
    end_times = {train_ids[0]: t0, val_ids[0]: t1, test_ids[0]: t2}
    counts = {gid: 2 for gid in end_times}
    store = MagicMock()
    store.fetch_account_game_end_times.return_value = end_times
    store.fetch_account_game_move_counts.return_value = counts
    store.fetch_for_game_ids.side_effect = lambda ids: [_datum(gid)[0] for gid in ids]
    split, got_times = load_account_split(store, "acct-1", limit=50)
    store.fetch_account_game_end_times.assert_called_once_with("acct-1")
    store.fetch_account_game_move_counts.assert_called_once_with("acct-1")
    store.fetch_for_account.assert_not_called()
    assert store.fetch_for_game_ids.call_count == 3
    assert list(store.fetch_for_game_ids.call_args_list[0].args[0]) == val_ids
    assert list(store.fetch_for_game_ids.call_args_list[1].args[0]) == test_ids
    assert list(store.fetch_for_game_ids.call_args_list[2].args[0]) == train_ids
    assert got_times == end_times
    assert split.salt == DEFAULT_SPLIT_SALT
    assert {d.game_id for d in split.train} == {train_ids[0]}
    assert {d.game_id for d in split.val} == {val_ids[0]}
    assert {d.game_id for d in split.test} == {test_ids[0]}


def test_load_account_split_limit_does_not_fetch_uncapped_train() -> None:
    train_ids, i = _ids_for_bucket(SplitBucket.TRAIN, 3)
    val_ids, i = _ids_for_bucket(SplitBucket.VAL, 1, start=i)
    test_ids, _i = _ids_for_bucket(SplitBucket.TEST, 1, start=i)
    t0 = datetime(2026, 5, 1, tzinfo=UTC)
    t1 = datetime(2026, 5, 2, tzinfo=UTC)
    t2 = datetime(2026, 5, 3, tzinfo=UTC)
    end_times = {
        train_ids[0]: t0,
        train_ids[1]: t1,
        train_ids[2]: t2,
        val_ids[0]: t2,
        test_ids[0]: t2,
    }
    n_moves = {gid: 8 for gid in end_times}
    store = MagicMock()
    store.fetch_account_game_end_times.return_value = end_times
    store.fetch_account_game_move_counts.return_value = n_moves
    fetched: list[list[str]] = []

    def _fetch(ids: list[str]) -> list[object]:
        ordered = list(ids)
        fetched.append(ordered)
        return [_datum(gid)[0] for gid in ordered]

    store.fetch_for_game_ids.side_effect = _fetch
    split, _end = load_account_split(store, "acct-1", limit=8)
    assert fetched[0] == val_ids
    assert fetched[1] == test_ids
    train_fetched = fetched[2]
    expected = cap_train_game_ids_by_game_id(train_ids, n_moves, 8)
    assert train_fetched == expected
    assert set(train_fetched).isdisjoint(set(val_ids) | set(test_ids))
    assert {d.game_id for d in split.val} == {val_ids[0]}
    assert {d.game_id for d in split.test} == {test_ids[0]}
    leftover = set(train_ids) - set(expected)
    assert leftover
    assert leftover.isdisjoint({d.game_id for d in split.train})


def test_cap_train_game_ids_keeps_whole_last_game() -> None:
    n_moves = {"g1": 8, "g1b": 8, "g2": 8}
    capped = cap_train_game_ids_by_game_id(["g1", "g1b", "g2"], n_moves, 10)
    assert capped == ["g1"]
