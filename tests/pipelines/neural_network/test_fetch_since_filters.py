"""extra_where is appended to count / fetch / boundary-expand SQL."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest

from chess_teacher.pipelines.neural_network import create_training_set as cts
from chess_teacher.pipelines.neural_network.create_training_set import (
    TrainingDataStore,
    account_id_in_sql,
)
from chess_teacher.pipelines.neural_network.models import PROCESSED_FLAG_PERSONAL


def _store() -> TrainingDataStore:
    store = TrainingDataStore(MagicMock())
    store._ensure_training_tables = lambda: None  # type: ignore[method-assign]
    store._datums_for_move_ids = lambda move_ids: []  # type: ignore[method-assign]
    return store


def test_store_has_no_hash_scan_or_count_since() -> None:
    src = inspect.getsource(cts)
    assert "hash_scan" not in src
    assert "HashScanCursor" not in src
    assert "count_hash_scan_pending" not in src
    assert "fetch_hash_scan_batch" not in src
    assert "count_since" not in src
    assert "count_new_moves_since" not in src
    assert "fetch_training_data_since" not in src
    assert "fetch_account_game_ingested_at" not in src
    assert not hasattr(TrainingDataStore, "count_since")
    assert not hasattr(TrainingDataStore, "count_hash_scan_pending")
    assert not hasattr(TrainingDataStore, "fetch_hash_scan_batch")
    assert "fetch_since" not in src
    assert not hasattr(TrainingDataStore, "fetch_since")


def test_fetch_account_game_move_counts_groups_by_game_id() -> None:
    store = _store()
    captured: list[str] = []

    def fake_query(sql: str, params: object) -> list[dict[str, object]]:
        captured.append(sql)
        return [{"game_id": "g1", "n_moves": 8}]

    store._query_moves_sql = fake_query  # type: ignore[method-assign]
    out = store.fetch_account_game_move_counts("acct-1")
    assert out == {"g1": 8}
    assert len(captured) == 1
    assert "COUNT(*)" in captured[0]
    assert "GROUP BY g.game_id" in captured[0]
    assert "g.account_id" in captured[0]


def test_count_unprocessed_train_filters_bucket_and_null_flag() -> None:
    store = _store()
    captured: list[str] = []

    def fake_query(sql: str, params: object) -> list[dict[str, int]]:
        captured.append(sql)
        return [{"n": 7}]

    store._query_moves_sql = fake_query  # type: ignore[method-assign]
    store._ensure_queue_tables = lambda: None  # type: ignore[method-assign]
    assert store.count_unprocessed_train(split_version="baseline-v1") == 7
    assert len(captured) == 1
    sql = captured[0]
    assert "gs.bucket = 'train'" in sql
    assert 'gs."already_processed_baseline" IS NULL' in sql
    assert "ml.game_split_assignments" in sql
    assert "end_time >" not in sql
    assert "ORDER BY" not in sql
    assert "games.raw_games" not in sql


def test_account_id_in_sql_builds_quoted_in_list() -> None:
    sql = account_id_in_sql(["b-2", "a-1", "b-2", ""])
    assert sql == "g.account_id IN ('a-1', 'b-2')"
    assert account_id_in_sql([]) == "1 = 0"
    assert account_id_in_sql(["", ""]) == "1 = 0"
    assert "O''Brien" in account_id_in_sql(["O'Brien"])


def test_count_unprocessed_train_includes_extra_where() -> None:
    store = _store()
    captured: list[str] = []

    def fake_query(sql: str, params: object) -> list[dict[str, int]]:
        captured.append(sql)
        return [{"n": 1}]

    store._query_moves_sql = fake_query  # type: ignore[method-assign]
    store._ensure_queue_tables = lambda: None  # type: ignore[method-assign]
    assert (
        store.count_unprocessed_train(
            split_version="baseline-v1",
            extra_where="g.account_id = 'acct-1'",
        )
        == 1
    )
    assert "g.account_id = 'acct-1'" in captured[0]


def test_fetch_unprocessed_train_batch_orders_and_expands_last_game() -> None:
    store = _store()
    captured: list[str] = []

    def fake_query(sql: str, params: object) -> list[dict[str, object]]:
        captured.append(sql)
        if "g.game_id = :boundary_game" in sql:
            return [
                {"move_id": "m1", "game_id": "gZ"},
                {"move_id": "m2", "game_id": "gZ"},
            ]
        return [
            {"move_id": "m0", "game_id": "gA"},
            {"move_id": "m1", "game_id": "gZ"},
        ] * 3

    store._query_moves_sql = fake_query  # type: ignore[method-assign]
    store._ensure_queue_tables = lambda: None  # type: ignore[method-assign]

    def fake_datums(move_ids: list[str]) -> list[MagicMock]:
        mapping = {"m0": "gA", "m1": "gZ", "m2": "gZ"}
        return [MagicMock(game_id=mapping[mid]) for mid in move_ids if mid in mapping]

    store._datums_for_move_ids = fake_datums  # type: ignore[method-assign]
    datums, game_ids = store.fetch_unprocessed_train_batch(
        split_version="baseline-v1",
        limit=5,
        extra_where="g.account_id = 'acct-1'",
    )
    assert [d.game_id for d in datums] == ["gA", "gA", "gA", "gZ", "gZ"]
    assert game_ids == ["gA", "gZ"]
    assert len(captured) == 2
    main_sql, expand_sql = captured
    assert "ORDER BY g.game_id ASC" in main_sql
    assert "gs.bucket = 'train'" in main_sql
    assert 'gs."already_processed_baseline" IS NULL' in main_sql
    assert "g.account_id = 'acct-1'" in main_sql
    assert "g.account_id = 'acct-1'" in expand_sql
    assert "g.game_id = :boundary_game" in expand_sql
    assert "end_time >" not in main_sql
    assert "games.raw_games" not in main_sql


def test_fetch_unprocessed_omits_games_with_zero_datums() -> None:
    store = _store()

    def fake_query(sql: str, params: object) -> list[dict[str, object]]:
        if "g.game_id = :boundary_game" in sql:
            return [{"move_id": "mZ", "game_id": "gZ"}]
        return [
            {"move_id": "mA", "game_id": "gA"},
            {"move_id": "mZ", "game_id": "gZ"},
        ]

    store._query_moves_sql = fake_query  # type: ignore[method-assign]
    store._ensure_queue_tables = lambda: None  # type: ignore[method-assign]
    store._datums_for_move_ids = lambda move_ids: [  # type: ignore[method-assign]
        MagicMock(game_id="gA") for mid in move_ids if mid == "mA"
    ]
    datums, game_ids = store.fetch_unprocessed_train_batch(
        split_version="baseline-v1",
        limit=10,
    )
    assert [d.game_id for d in datums] == ["gA"]
    assert game_ids == ["gA"]


def test_fetch_unprocessed_train_rejects_unknown_flag() -> None:
    store = _store()
    store._ensure_queue_tables = lambda: None  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="flag_column"):
        store.count_unprocessed_train(split_version="v1", flag_column="not_a_flag")


def test_fetch_unprocessed_train_personal_flag_and_in_list() -> None:
    store = _store()
    captured: list[str] = []

    def fake_query(sql: str, params: object) -> list[dict[str, object]]:
        captured.append(sql)
        return []

    store._query_moves_sql = fake_query  # type: ignore[method-assign]
    store._ensure_queue_tables = lambda: None  # type: ignore[method-assign]
    extra = account_id_in_sql(["acct-a", "acct-b"])
    datums, game_ids = store.fetch_unprocessed_train_batch(
        split_version="baseline-v1",
        limit=10,
        flag_column=PROCESSED_FLAG_PERSONAL,
        extra_where=extra,
    )
    assert datums == []
    assert game_ids == []
    assert len(captured) == 1
    sql = captured[0]
    assert 'gs."already_processed_personal" IS NULL' in sql
    assert "already_processed_baseline" not in sql
    assert "gs.bucket = 'train'" in sql
    assert extra in sql
    assert "gs.bucket = 'val'" not in sql


def test_fetch_registry_bucket_batch_orders_and_expands_last_game() -> None:
    store = _store()
    captured: list[str] = []

    def fake_query(sql: str, params: object) -> list[dict[str, object]]:
        captured.append(sql)
        if "g.game_id = :boundary_game" in sql:
            return [
                {"move_id": "m1", "game_id": "gZ"},
                {"move_id": "m2", "game_id": "gZ"},
            ]
        return [
            {"move_id": "m0", "game_id": "gA"},
            {"move_id": "m1", "game_id": "gZ"},
        ] * 3

    store._query_moves_sql = fake_query  # type: ignore[method-assign]
    store._ensure_queue_tables = lambda: None  # type: ignore[method-assign]

    def fake_datums(move_ids: list[str]) -> list[MagicMock]:
        mapping = {"m0": "gA", "m1": "gZ", "m2": "gZ"}
        return [MagicMock(game_id=mapping[mid]) for mid in move_ids if mid in mapping]

    store._datums_for_move_ids = fake_datums  # type: ignore[method-assign]
    datums = store.fetch_registry_bucket_batch(
        split_version="baseline-v1",
        bucket="val",
        limit=5,
        extra_where="g.account_id = 'acct-1'",
    )
    assert [d.game_id for d in datums] == ["gA", "gA", "gA", "gZ", "gZ"]
    assert len(captured) == 2
    main_sql, expand_sql = captured
    assert "ORDER BY g.game_id ASC" in main_sql
    assert "gs.bucket = :bucket" in main_sql
    assert "already_processed" not in main_sql
    assert "g.account_id = 'acct-1'" in main_sql
    assert "g.game_id = :boundary_game" in expand_sql
    assert "end_time >" not in main_sql


def test_fetch_split_val_datums_scopes_val_bucket_and_accounts() -> None:
    store = _store()
    captured: list[str] = []

    def fake_query(sql: str, params: object) -> list[dict[str, object]]:
        captured.append(sql)
        return [{"move_id": "m1", "game_id": "v1"}]

    store._query_moves_sql = fake_query  # type: ignore[method-assign]
    store._ensure_queue_tables = lambda: None  # type: ignore[method-assign]
    extra = account_id_in_sql(["acct-1"])
    datums = store.fetch_split_val_datums(split_version="baseline-v1", extra_where=extra)
    assert datums == []
    assert len(captured) == 1
    sql = captured[0]
    assert "gs.bucket = :bucket" in sql
    assert extra in sql
    assert "LIMIT" not in sql
    assert "already_processed" not in sql
