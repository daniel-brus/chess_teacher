"""extra_where is appended to count / fetch / boundary-expand SQL."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from chess_teacher.pipelines.neural_network.create_training_set import (
    TrainingDataStore,
    count_new_moves_since,
    fetch_training_data_since,
)

_EXCLUDE = (
    "NOT EXISTS (SELECT 1 FROM ml.game_split_assignments gs WHERE gs.bucket IN ('val', 'test'))"
)


def _store() -> TrainingDataStore:
    store = TrainingDataStore(MagicMock())
    store._ensure_training_tables = lambda: None  # type: ignore[method-assign]
    store._datums_for_move_ids = lambda move_ids: []  # type: ignore[method-assign]
    return store


def test_count_since_includes_extra_where() -> None:
    store = _store()
    captured: list[str] = []

    def fake_query(sql: str, params: object) -> list[dict[str, int]]:
        captured.append(sql)
        return [{"n": 0}]

    store._query_moves_sql = fake_query  # type: ignore[method-assign]
    assert store.count_since(None, extra_where=_EXCLUDE) == 0
    assert len(captured) == 1
    assert _EXCLUDE in captured[0]
    assert "AND (" in captured[0]


def test_count_since_none_extra_where_unchanged() -> None:
    store = _store()
    captured: list[str] = []

    def fake_query(sql: str, params: object) -> list[dict[str, int]]:
        captured.append(sql)
        return [{"n": 3}]

    store._query_moves_sql = fake_query  # type: ignore[method-assign]
    assert store.count_since(None) == 3
    assert "AND (" not in captured[0]


def test_fetch_since_includes_extra_where_on_main_query() -> None:
    store = _store()
    captured: list[str] = []
    t0 = datetime(2026, 1, 1, tzinfo=UTC)

    def fake_query(sql: str, params: object) -> list[dict[str, object]]:
        captured.append(sql)
        return [{"move_id": "m1", "end_time": t0}]

    store._query_moves_sql = fake_query  # type: ignore[method-assign]
    datums, max_t = store.fetch_since(None, limit=10, extra_where=_EXCLUDE)
    assert datums == []
    assert max_t == t0
    assert len(captured) == 1
    assert _EXCLUDE in captured[0]


def test_fetch_since_includes_extra_where_on_boundary_expand() -> None:
    store = _store()
    captured: list[str] = []
    t0 = datetime(2026, 1, 2, tzinfo=UTC)

    def fake_query(sql: str, params: object) -> list[dict[str, object]]:
        captured.append(sql)
        if "end_time = :boundary" in sql:
            return [
                {"move_id": "m1", "end_time": t0},
                {"move_id": "m2", "end_time": t0},
            ]
        return [{"move_id": "m1", "end_time": t0}] * 5

    store._query_moves_sql = fake_query  # type: ignore[method-assign]
    store.fetch_since(None, limit=5, extra_where=_EXCLUDE)
    assert len(captured) == 2
    main_sql, expand_sql = captured
    assert _EXCLUDE in main_sql
    assert _EXCLUDE in expand_sql
    assert "end_time = :boundary" in expand_sql


def test_wrappers_forward_extra_where(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class FakeStore:
        def __init__(self, db_client: object) -> None:
            seen["db"] = db_client

        def count_since(self, cutoff: object, *, extra_where: str | None = None) -> int:
            seen["count_extra"] = extra_where
            return 1

        def fetch_since(
            self,
            cutoff: object,
            *,
            limit: int | None = None,
            extra_where: str | None = None,
        ) -> tuple[list[object], None]:
            seen["fetch_extra"] = extra_where
            seen["limit"] = limit
            return [], None

    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.create_training_set.TrainingDataStore",
        FakeStore,
    )
    assert count_new_moves_since(None, extra_where=_EXCLUDE) == 1
    assert seen["count_extra"] == _EXCLUDE
    fetch_training_data_since(None, limit=7, extra_where=_EXCLUDE)
    assert seen["fetch_extra"] == _EXCLUDE
    assert seen["limit"] == 7
