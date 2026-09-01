"""Game-level train / val / test splits for offline baseline experiments.

Splits by ``game_id`` so moves from the same game never leak across buckets.
Deterministic via MD5 hash — stable across runs and processes (unlike ``hash()``).

See ``.agents/docs/ml-training-roadmap.md`` Phase 1.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from chess_teacher.pipelines.neural_network.ply_weights import user_not_sf_best_mask

if TYPE_CHECKING:
    from chess_teacher.pipelines.neural_network.create_training_set import TrainingDatum

DEFAULT_SPLIT_SALT = "baseline-v1"
TRAIN_BUCKET_MAX = 84  # 0-84 inclusive -> 85%
VAL_BUCKET_MAX = 94  # 85-94 inclusive -> 10%
# 95-99 -> test (5%)


class SplitBucket(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass(frozen=True)
class SplitCounts:
    """Move and game counts for one split bucket."""

    bucket: SplitBucket
    n_games: int
    n_moves: int
    sf_disagree_frac: float | None = None


@dataclass(frozen=True)
class GameSplitResult:
    """Datums partitioned by game; all moves from a game share one bucket."""

    train: tuple[TrainingDatum, ...]
    val: tuple[TrainingDatum, ...]
    test: tuple[TrainingDatum, ...]
    salt: str
    counts: tuple[SplitCounts, ...]

    @property
    def train_datums(self) -> list[TrainingDatum]:
        return list(self.train)

    @property
    def val_datums(self) -> list[TrainingDatum]:
        return list(self.val)

    @property
    def test_datums(self) -> list[TrainingDatum]:
        return list(self.test)


def game_split_bucket(game_id: str, *, salt: str = DEFAULT_SPLIT_SALT) -> SplitBucket:
    """Map one game to train / val / test via ``md5(game_id:salt) % 100``."""
    digest = hashlib.md5(f"{game_id}:{salt}".encode()).hexdigest()
    bucket = int(digest, 16) % 100
    if bucket <= TRAIN_BUCKET_MAX:
        return SplitBucket.TRAIN
    if bucket <= VAL_BUCKET_MAX:
        return SplitBucket.VAL
    return SplitBucket.TEST


def _disagree_fraction(datums: list[TrainingDatum]) -> float | None:
    """SF-disagree fraction among datums with usable candidate targets."""
    from chess_teacher.pipelines.neural_network.create_training_set import TrainingBatch

    if not datums:
        return None
    batch = TrainingBatch(datums)
    feats, _mask, labels, kept = batch.candidate_style_targets()
    if not kept:
        return None
    disagree = user_not_sf_best_mask(feats, labels)
    return float(np.mean(disagree))


def split_datums_by_game(
    datums: list[TrainingDatum],
    *,
    salt: str = DEFAULT_SPLIT_SALT,
    compute_disagree_frac: bool = True,
    bucket_for_game: Callable[[str], SplitBucket] | None = None,
) -> GameSplitResult:
    """Partition datums by ``game_id``; every move from a game stays in one bucket."""
    if bucket_for_game is None:
        def bucket_for_game(gid: str) -> SplitBucket:
            return game_split_bucket(gid, salt=salt)
    return _partition_datums(
        datums,
        bucket_for_game=bucket_for_game,
        salt=salt,
        compute_disagree_frac=compute_disagree_frac,
    )


def _partition_datums(
    datums: list[TrainingDatum],
    *,
    bucket_for_game: Callable[[str], SplitBucket],
    salt: str,
    compute_disagree_frac: bool,
) -> GameSplitResult:
    by_game: dict[str, list[TrainingDatum]] = {}
    for d in datums:
        by_game.setdefault(d.game_id, []).append(d)

    train: list[TrainingDatum] = []
    val: list[TrainingDatum] = []
    test: list[TrainingDatum] = []
    train_games = val_games = test_games = 0

    for game_id, game_datums in by_game.items():
        bucket = bucket_for_game(game_id)
        if bucket is SplitBucket.TRAIN:
            train.extend(game_datums)
            train_games += 1
        elif bucket is SplitBucket.VAL:
            val.extend(game_datums)
            val_games += 1
        else:
            test.extend(game_datums)
            test_games += 1

    def _counts(
        bucket: SplitBucket,
        moves: list[TrainingDatum],
        n_games: int,
    ) -> SplitCounts:
        frac = _disagree_fraction(moves) if compute_disagree_frac else None
        return SplitCounts(
            bucket=bucket,
            n_games=n_games,
            n_moves=len(moves),
            sf_disagree_frac=frac,
        )

    return GameSplitResult(
        train=tuple(train),
        val=tuple(val),
        test=tuple(test),
        salt=salt,
        counts=(
            _counts(SplitBucket.TRAIN, train, train_games),
            _counts(SplitBucket.VAL, val, val_games),
            _counts(SplitBucket.TEST, test, test_games),
        ),
    )
