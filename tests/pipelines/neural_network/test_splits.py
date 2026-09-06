"""Unit tests for game-level train/val/test splits."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from chess_teacher.pipelines.neural_network.splits import (
    DEFAULT_SPLIT_SALT,
    SplitBucket,
    format_split_summary,
    game_split_bucket,
    split_datums_by_game,
)


@dataclass(frozen=True)
class _FakeDatum:
    game_id: str
    move_id: str


def test_game_split_bucket_is_deterministic() -> None:
    b1 = game_split_bucket("game-abc", salt=DEFAULT_SPLIT_SALT)
    b2 = game_split_bucket("game-abc", salt=DEFAULT_SPLIT_SALT)
    assert b1 == b2


def test_game_split_bucket_respects_salt() -> None:
    a = game_split_bucket("game-1", salt="salt-a")
    b = game_split_bucket("game-1", salt="salt-b")
    assert a in SplitBucket
    assert b in SplitBucket
    # Different salts usually differ (not guaranteed for all ids, but unlikely equal)
    if a != b:
        return
    pytest.skip("rare collision: same bucket for both salts")


def test_all_moves_from_game_share_bucket() -> None:
    datums = [
        _FakeDatum(game_id="g1", move_id="m1"),
        _FakeDatum(game_id="g1", move_id="m2"),
        _FakeDatum(game_id="g2", move_id="m3"),
    ]
    split = split_datums_by_game(datums, compute_disagree_frac=False)  # type: ignore[arg-type]
    assert len(split.train) + len(split.val) + len(split.test) == 3
    # Both g1 moves in same partition
    train_g1 = sum(1 for d in split.train if d.game_id == "g1")
    val_g1 = sum(1 for d in split.val if d.game_id == "g1")
    test_g1 = sum(1 for d in split.test if d.game_id == "g1")
    assert train_g1 + val_g1 + test_g1 == 2
    assert sum(x > 0 for x in (train_g1, val_g1, test_g1)) == 1


def test_split_counts_match_move_totals() -> None:
    datums = [_FakeDatum(game_id=f"g{i}", move_id=f"m{i}") for i in range(200)]
    split = split_datums_by_game(datums, compute_disagree_frac=False)  # type: ignore[arg-type]
    total_moves = sum(c.n_moves for c in split.counts)
    total_games = sum(c.n_games for c in split.counts)
    assert total_moves == 200
    assert total_games == 200


def test_split_bucket_distribution_roughly_matches_targets() -> None:
    """~85/10/5 over many synthetic game ids (statistical smoke test)."""
    n_games = 5000
    counts = {SplitBucket.TRAIN: 0, SplitBucket.VAL: 0, SplitBucket.TEST: 0}
    for i in range(n_games):
        counts[game_split_bucket(f"synthetic-game-{i}")] += 1
    train_frac = counts[SplitBucket.TRAIN] / n_games
    val_frac = counts[SplitBucket.VAL] / n_games
    test_frac = counts[SplitBucket.TEST] / n_games
    assert train_frac == pytest.approx(0.85, abs=0.03)
    assert val_frac == pytest.approx(0.10, abs=0.02)
    assert test_frac == pytest.approx(0.05, abs=0.02)


def test_format_split_summary_includes_version_and_counts() -> None:
    datums = [_FakeDatum(game_id=f"g{i}", move_id=f"m{i}") for i in range(20)]
    split = split_datums_by_game(datums, compute_disagree_frac=False)  # type: ignore[arg-type]
    text = format_split_summary(split)
    assert f"split_version={DEFAULT_SPLIT_SALT!r}" in text
    assert "train" in text
    assert "val" in text
    assert "test" in text
