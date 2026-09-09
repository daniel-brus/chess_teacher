"""Account-level user split: same hash buckets as platform baseline.

Uses ``game_split_bucket(game_id, salt=DEFAULT_SPLIT_SALT)`` (``baseline-v1``,
85/10/5). Does not write the split registry.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING

from chess_teacher.pipelines.neural_network.splits import (
    DEFAULT_SPLIT_SALT,
    GameSplitResult,
    SplitBucket,
    SplitCounts,
    game_split_bucket,
    split_datums_by_game,
)

if TYPE_CHECKING:
    from chess_teacher.pipelines.neural_network.create_training_set import (
        TrainingDataStore,
        TrainingDatum,
    )


DEFAULT_MIN_USER_TRAIN_MOVES = 300
DEFAULT_MIN_USER_VAL_MOVES = 10


def partition_account_game_ids(
    game_ids: Iterable[str],
    *,
    salt: str = DEFAULT_SPLIT_SALT,
) -> tuple[list[str], list[str], list[str]]:
    """Return ``(train_ids, val_ids, test_ids)`` via ``game_split_bucket``."""
    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []
    for gid in game_ids:
        if not gid:
            continue
        bucket = game_split_bucket(gid, salt=salt)
        if bucket is SplitBucket.TRAIN:
            train_ids.append(gid)
        elif bucket is SplitBucket.VAL:
            val_ids.append(gid)
        else:
            test_ids.append(gid)
    return train_ids, val_ids, test_ids


def cap_train_game_ids_by_game_id(
    train_ids: Sequence[str],
    n_moves_by_game: Mapping[str, int],
    limit: int,
) -> list[str]:
    """Complete train games in sorted ``game_id`` order until move sum would exceed ``limit``.

    Does not split the last included game. First game is always taken even if it
    alone exceeds ``limit``.
    """
    selected: list[str] = []
    running = 0
    cap = int(limit)
    for gid in sorted(gid for gid in train_ids if gid):
        n = int(n_moves_by_game.get(gid, 0))
        if n <= 0:
            continue
        if selected and running + n > cap:
            break
        selected.append(gid)
        running += n
    return selected


def _split_result(
    train: Sequence[TrainingDatum],
    val: Sequence[TrainingDatum],
    test: Sequence[TrainingDatum] = (),
    *,
    salt: str = DEFAULT_SPLIT_SALT,
) -> GameSplitResult:
    train_ids = list(dict.fromkeys(d.game_id for d in train))
    val_ids = list(dict.fromkeys(d.game_id for d in val))
    test_ids = list(dict.fromkeys(d.game_id for d in test))
    return GameSplitResult(
        train=tuple(train),
        val=tuple(val),
        test=tuple(test),
        salt=salt,
        counts=(
            SplitCounts(
                bucket=SplitBucket.TRAIN,
                n_games=len(train_ids),
                n_moves=len(train),
            ),
            SplitCounts(
                bucket=SplitBucket.VAL,
                n_games=len(val_ids),
                n_moves=len(val),
            ),
            SplitCounts(
                bucket=SplitBucket.TEST,
                n_games=len(test_ids),
                n_moves=len(test),
            ),
        ),
    )


def split_account_datums(
    datums: Sequence[TrainingDatum],
    *,
    salt: str = DEFAULT_SPLIT_SALT,
) -> GameSplitResult:
    """Partition one account with the same hash rule as ``game_split_bucket``."""
    return split_datums_by_game(list(datums), salt=salt, compute_disagree_frac=False)


def load_account_split(
    store: TrainingDataStore,
    account_id: str,
    *,
    limit: int | None = None,
    salt: str = DEFAULT_SPLIT_SALT,
) -> tuple[GameSplitResult, dict[str, datetime]]:
    """Hash-split eligible games, then fetch moves per bucket.

    ``limit`` caps **train** game ids before hydrate (sorted ``game_id``,
    complete games). Val and test are never ply-truncated. Does not
    call ``fetch_for_account``.
    """
    end_times = store.fetch_account_game_end_times(account_id)
    n_moves = store.fetch_account_game_move_counts(account_id)
    train_ids, val_ids, test_ids = partition_account_game_ids(end_times, salt=salt)
    val = store.fetch_for_game_ids(val_ids)
    test = store.fetch_for_game_ids(test_ids)
    if limit is not None:
        train_ids = cap_train_game_ids_by_game_id(train_ids, n_moves, int(limit))
    train = store.fetch_for_game_ids(train_ids)
    return _split_result(train, val, test, salt=salt), end_times
