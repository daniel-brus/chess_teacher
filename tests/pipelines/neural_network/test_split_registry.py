"""Unit tests for persistent split registry."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from chess_teacher.pipelines.neural_network.models import GameSplitAssignment
from chess_teacher.pipelines.neural_network.split_registry import (
    SplitRegistry,
    clear_personal_processed,
)
from chess_teacher.pipelines.neural_network.splits import (
    DEFAULT_SPLIT_SALT,
    SplitBucket,
    game_split_bucket,
)
from chess_teacher.utils.db.client import WriteResult, WriteStrategy


@dataclass(frozen=True)
class _FakeDatum:
    game_id: str
    move_id: str


def test_game_split_assignment_metadata_in_sync() -> None:
    errors = GameSplitAssignment.validate_metadata_sync()
    assert not errors, "\n  ".join(errors)


def test_exclude_holdout_sql_contains_version_and_buckets() -> None:
    db = MagicMock()
    registry = SplitRegistry(db, split_version="baseline-v1")
    sql = registry.exclude_holdout_games_sql(game_id_column="g.game_id")
    assert "ml.game_split_assignments" in sql
    assert "'baseline-v1'" in sql
    assert "g.game_id" in sql
    assert "'val'" in sql
    assert "'test'" in sql


def test_ensure_games_inserts_expected_bucket() -> None:
    db = MagicMock()
    db.insert.return_value = WriteResult(
        strategy=WriteStrategy.INSERT_IGNORE,
        rows_inserted=2,
    )
    registry = SplitRegistry(db, split_version=DEFAULT_SPLIT_SALT)

    with patch.object(
        GameSplitAssignment,
        "fetch_all_from_db",
        return_value=[],
    ):
        n = registry.ensure_games(["game-a", "game-b"])

    assert n == 2
    db.ensure_metadata.assert_called()
    insert_call = db.insert.call_args
    records = insert_call[0][0]
    assert len(records) == 2
    assert records[0]["split_version"] == DEFAULT_SPLIT_SALT
    assert records[0]["bucket"] == game_split_bucket("game-a").value
    assert records[1]["bucket"] == game_split_bucket("game-b").value


def test_ensure_games_skips_existing() -> None:
    db = MagicMock()
    registry = SplitRegistry(db, split_version=DEFAULT_SPLIT_SALT)
    existing = GameSplitAssignment(
        split_version=DEFAULT_SPLIT_SALT,
        game_id="game-a",
        bucket=SplitBucket.TRAIN.value,
        assigned_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    with patch.object(
        GameSplitAssignment,
        "fetch_all_from_db",
        return_value=[existing],
    ):
        n = registry.ensure_games(["game-a"])

    assert n == 0
    db.insert.assert_not_called()


def test_split_datums_uses_registry_buckets() -> None:
    db = MagicMock()
    registry = SplitRegistry(db, split_version=DEFAULT_SPLIT_SALT)
    datums = [
        _FakeDatum(game_id="g1", move_id="m1"),
        _FakeDatum(game_id="g1", move_id="m2"),
        _FakeDatum(game_id="g2", move_id="m3"),
    ]

    def fake_buckets(game_ids: list[str]) -> dict[str, SplitBucket]:
        return {gid: game_split_bucket(gid) for gid in game_ids}

    with (
        patch.object(SplitRegistry, "ensure_games", return_value=0) as ensure,
        patch.object(SplitRegistry, "fetch_buckets", side_effect=fake_buckets),
    ):
        split = registry.split_datums(datums, compute_disagree_frac=False)  # type: ignore[arg-type]

    ensure.assert_called_once()
    assert len(split.train) + len(split.val) + len(split.test) == 3
    g1_bucket = game_split_bucket("g1")
    g1_in_train = sum(1 for d in split.train if d.game_id == "g1")
    g1_in_val = sum(1 for d in split.val if d.game_id == "g1")
    g1_in_test = sum(1 for d in split.test if d.game_id == "g1")
    assert g1_in_train + g1_in_val + g1_in_test == 2
    if g1_bucket is SplitBucket.TRAIN:
        assert g1_in_train == 2
    elif g1_bucket is SplitBucket.VAL:
        assert g1_in_val == 2
    else:
        assert g1_in_test == 2


def test_split_datums_assign_if_missing_false_skips_ensure() -> None:
    db = MagicMock()
    registry = SplitRegistry(db, split_version=DEFAULT_SPLIT_SALT)
    datums = [_FakeDatum(game_id="g1", move_id="m1")]

    with (
        patch.object(SplitRegistry, "ensure_games", return_value=0) as ensure,
        patch.object(
            SplitRegistry,
            "fetch_buckets",
            return_value={"g1": SplitBucket.TRAIN},
        ),
    ):
        split = registry.split_datums(
            datums,  # type: ignore[arg-type]
            assign_if_missing=False,
            compute_disagree_frac=False,
        )

    ensure.assert_not_called()
    assert len(split.train) == 1


def test_fetch_game_ids_for_bucket_uses_version_and_bucket() -> None:
    db = MagicMock()
    registry = SplitRegistry(db, split_version="baseline-v1")
    rows = [
        GameSplitAssignment(
            split_version="baseline-v1",
            game_id="g-val-1",
            bucket=SplitBucket.VAL.value,
            assigned_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
    ]
    with patch.object(GameSplitAssignment, "fetch_all_from_db", return_value=rows) as fetch:
        ids = registry.fetch_game_ids_for_bucket(SplitBucket.VAL)
    assert ids == ["g-val-1"]
    where = fetch.call_args.kwargs["where"]
    assert "baseline-v1" in where
    assert "val" in where


def test_eligible_sql_filters_account_when_set() -> None:
    db = MagicMock()
    registry = SplitRegistry(db, split_version=DEFAULT_SPLIT_SALT)
    sql, params = registry._eligible_from_sql(account_id="acct-1")
    assert "g.account_id = :account_id" in sql
    assert params["account_id"] == "acct-1"
    sql_all, params_all = registry._eligible_from_sql()
    assert "account_id" not in sql_all
    assert params_all == {}


def test_ensure_eligible_games_for_account_requires_id() -> None:
    db = MagicMock()
    registry = SplitRegistry(db, split_version=DEFAULT_SPLIT_SALT)
    with pytest.raises(ValueError, match="account_id is required"):
        registry.ensure_eligible_games_for_account("")


def test_mark_processed_updates_train_null_flag_only() -> None:
    db = MagicMock()
    db.update_where.return_value = 2
    registry = SplitRegistry(db, split_version="baseline-v1")
    n = registry.mark_processed(["g2", "g1"])
    assert n == 2
    values, where = db.update_where.call_args.args[1], db.update_where.call_args.args[2]
    assert "already_processed_baseline" in values
    assert values["already_processed_baseline"] is not None
    assert "baseline-v1" in where
    assert "train" in where
    assert "already_processed_baseline" in where
    assert "IS NULL" in where
    assert "g1" in where
    assert "g2" in where


def test_mark_processed_empty_is_noop() -> None:
    db = MagicMock()
    registry = SplitRegistry(db, split_version=DEFAULT_SPLIT_SALT)
    assert registry.mark_processed([]) == 0
    db.update_where.assert_not_called()


def test_clear_processed_nulls_flag() -> None:
    db = MagicMock()
    db.update_where.return_value = 4
    registry = SplitRegistry(db, split_version="baseline-v1")
    n = registry.clear_processed()
    assert n == 4
    values, where = db.update_where.call_args.args[1], db.update_where.call_args.args[2]
    assert values["already_processed_baseline"] is None
    assert "baseline-v1" in where


def test_clear_personal_processed_scopes_game_ids() -> None:
    db = MagicMock()
    db.update_where.return_value = 2
    n = clear_personal_processed(db, ["g2", "g1"], split_version="baseline-v1")
    assert n == 2
    values, where = db.update_where.call_args.args[1], db.update_where.call_args.args[2]
    assert values["already_processed_personal"] is None
    assert "already_processed_baseline" not in values
    assert "baseline-v1" in where
    assert "g1" in where
    assert "g2" in where


def test_clear_personal_processed_empty_is_noop() -> None:
    db = MagicMock()
    assert clear_personal_processed(db, []) == 0
    db.update_where.assert_not_called()


def test_ensure_games_inserts_null_processed_flags() -> None:
    db = MagicMock()
    db.insert.return_value = WriteResult(
        strategy=WriteStrategy.INSERT_IGNORE,
        rows_inserted=1,
    )
    registry = SplitRegistry(db, split_version=DEFAULT_SPLIT_SALT)

    with patch.object(GameSplitAssignment, "fetch_all_from_db", return_value=[]):
        registry.ensure_games(["game-a"])

    records = db.insert.call_args[0][0]
    assert records[0]["already_processed_baseline"] is None
    assert records[0]["already_processed_personal"] is None
