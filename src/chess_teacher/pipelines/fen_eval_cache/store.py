"""Durable + in-memory stores for the EPD eval cache."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from chess_teacher.pipelines.fen_eval_cache.tables import PositionEvalRow
from chess_teacher.utils.db.client import DatabaseClient, MergeStrategy
from chess_teacher.utils.general_utils import get_current_datetime, quote_ident, quote_literal
from chess_teacher.utils.logging import get_logger

logger = get_logger()


@dataclass(frozen=True, slots=True)
class StoredEval:
    epd: str
    engine: str
    eval_white_pov: float
    eval_depth: int
    candidates: dict[str, Any]
    candidate_nodes: int


class EvalStore(Protocol):
    def get_many(self, epds: list[str]) -> dict[str, StoredEval]: ...

    def upsert_many(self, rows: list[StoredEval]) -> None: ...

    def touch(self, epds: list[str]) -> None: ...

    def evict(self, capacity: int) -> int: ...


class MemoryEvalStore:
    """Process-local LRU used as a front and as a DB fallback."""

    def __init__(self, *, capacity: int | None = None) -> None:
        self._rows: OrderedDict[str, StoredEval] = OrderedDict()
        self.capacity = capacity

    def get_many(self, epds: list[str]) -> dict[str, StoredEval]:
        found: dict[str, StoredEval] = {}
        for epd in epds:
            row = self._rows.get(epd)
            if row is not None:
                self._rows.move_to_end(epd)
                found[epd] = row
        return found

    def upsert_many(self, rows: list[StoredEval]) -> None:
        for row in rows:
            self._rows[row.epd] = row
            self._rows.move_to_end(row.epd)
        if self.capacity is not None:
            self.evict(self.capacity)

    def touch(self, epds: list[str]) -> None:
        for epd in epds:
            if epd in self._rows:
                self._rows.move_to_end(epd)

    def evict(self, capacity: int) -> int:
        removed = 0
        while len(self._rows) > capacity:
            self._rows.popitem(last=False)
            removed += 1
        return removed


class PostgresEvalStore:
    def __init__(self, db_client: DatabaseClient) -> None:
        self._db = db_client
        self._meta = PositionEvalRow.get_metadata()
        self._db.ensure_metadata(self._meta)

    def get_many(self, epds: list[str]) -> dict[str, StoredEval]:
        if not epds:
            return {}
        table = self._meta.qualified_name_sql()
        in_list = ", ".join(quote_literal(epd) for epd in epds)
        sql = (
            f"SELECT epd, engine, eval_white_pov, eval_depth, candidates, candidate_nodes\n"
            f"FROM {table}\n"
            f"WHERE {quote_ident('epd')} IN ({in_list});"
        )
        found: dict[str, StoredEval] = {}
        for raw in self._db.engine.execute_parameterized_query(sql, {}):
            payload = raw["candidates"]
            if not isinstance(payload, dict):
                continue
            found[str(raw["epd"])] = StoredEval(
                epd=str(raw["epd"]),
                engine=str(raw["engine"]),
                eval_white_pov=float(raw["eval_white_pov"]),
                eval_depth=int(raw["eval_depth"]),
                candidates=payload,
                candidate_nodes=int(raw["candidate_nodes"]),
            )
        return found

    def upsert_many(self, rows: list[StoredEval]) -> None:
        if not rows:
            return
        now = get_current_datetime()
        records = [
            {
                "epd": row.epd,
                "engine": row.engine,
                "eval_white_pov": row.eval_white_pov,
                "eval_depth": row.eval_depth,
                "candidates": row.candidates,
                "candidate_nodes": row.candidate_nodes,
                "last_used": now,
            }
            for row in rows
        ]
        self._db.merge(records, self._meta, strategy=MergeStrategy.upsert())

    def touch(self, epds: list[str]) -> None:
        if not epds:
            return
        table = self._meta.qualified_name_sql()
        in_list = ", ".join(quote_literal(epd) for epd in epds)
        now = quote_literal(_iso(get_current_datetime()))
        sql = (
            f"UPDATE {table}\n"
            f"SET {quote_ident('last_used')} = {now}::timestamptz\n"
            f"WHERE {quote_ident('epd')} IN ({in_list});"
        )
        self._db.engine.execute_write(sql, {})

    def evict(self, capacity: int) -> int:
        if capacity < 0:
            return 0
        table = self._meta.qualified_name_sql()
        count_sql = f"SELECT COUNT(*) AS n FROM {table};"
        count_rows = self._db.engine.execute_parameterized_query(count_sql, {})
        count = int(count_rows[0]["n"]) if count_rows else 0
        extra = count - capacity
        if extra <= 0:
            return 0
        sql = (
            f"DELETE FROM {table}\n"
            f"WHERE {quote_ident('epd')} IN (\n"
            f"  SELECT {quote_ident('epd')} FROM {table}\n"
            f"  ORDER BY {quote_ident('last_used')} ASC\n"
            f"  LIMIT {int(extra)}\n"
            f");"
        )
        return self._db.engine.execute_write(sql, {})


def _iso(value: datetime) -> str:
    return value.isoformat()
