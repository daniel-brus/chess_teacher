"""Persistent game split assignments (Phase 1b).

Stores ``(split_version, game_id) → bucket`` in ``ml.game_split_assignments``.
Assignment rule matches ``splits.game_split_bucket`` — registry is persistence only.

See ``.agents/docs/ml-training-roadmap.md`` Phase 1b.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from chess_teacher.pipelines.neural_network.create_training_set import TrainingDatum
from chess_teacher.pipelines.neural_network.models import GameSplitAssignment
from chess_teacher.pipelines.neural_network.splits import (
    DEFAULT_SPLIT_SALT,
    GameSplitResult,
    SplitBucket,
    game_split_bucket,
    split_datums_by_game,
)
from chess_teacher.pipelines.preprocessing.games import Game
from chess_teacher.pipelines.preprocessing.moves import Move, MoveCharacteristics
from chess_teacher.utils.db.client import DatabaseClient, get_db_client
from chess_teacher.utils.general_utils import generate_ident_is_literal, quote_literal
from chess_teacher.utils.logging import get_logger

logger = get_logger()

# Same eligibility as TrainingDataStore (moves with SF candidate evals + end_time).
_ELIGIBLE_GAMES_SQL = """
            FROM games.moves m
            INNER JOIN games.games g ON g.game_id = m.game_id
            INNER JOIN games.move_characteristics mc ON mc.move_id = m.move_id
            WHERE g.end_time IS NOT NULL
              AND mc.candidate_evaluations IS NOT NULL
"""

_MOVES_QUERY_SESSION_SETTINGS = {"max_parallel_workers_per_gather": "0"}


@dataclass(frozen=True)
class BackfillResult:
    split_version: str
    eligible_games: int
    newly_assigned: int
    already_assigned: int


@dataclass(frozen=True)
class SplitRegistry:
    """Read/write persistent split assignments for one ``split_version`` (salt)."""

    db_client: DatabaseClient
    split_version: str = DEFAULT_SPLIT_SALT

    def ensure_table(self) -> None:
        self.db_client.ensure_metadata(GameSplitAssignment.get_metadata())

    def bucket_for_game(self, game_id: str, *, assign_if_missing: bool = True) -> SplitBucket:
        """Return bucket for ``game_id``; optionally persist a new assignment."""
        self.ensure_table()
        rows = GameSplitAssignment.fetch_all_from_db(
            self.db_client,
            where=self._where_game(game_id),
            limit=1,
        )
        if rows:
            return SplitBucket(rows[0].bucket)
        if not assign_if_missing:
            return game_split_bucket(game_id, salt=self.split_version)
        self.ensure_games([game_id])
        rows = GameSplitAssignment.fetch_all_from_db(
            self.db_client,
            where=self._where_game(game_id),
            limit=1,
        )
        if not rows:
            raise RuntimeError(f"Failed to assign split for game_id={game_id!r}")
        return SplitBucket(rows[0].bucket)

    def ensure_games(self, game_ids: Iterable[str]) -> int:
        """Insert missing assignments; return count of newly written rows."""
        unique = sorted({gid for gid in game_ids if gid})
        if not unique:
            return 0
        self.ensure_table()
        existing = self.fetch_buckets(unique)
        from chess_teacher.utils.general_utils import get_current_datetime

        assigned_at = get_current_datetime()
        pending: list[GameSplitAssignment] = []
        for game_id in unique:
            if game_id in existing:
                continue
            bucket = game_split_bucket(game_id, salt=self.split_version)
            pending.append(
                GameSplitAssignment(
                    split_version=self.split_version,
                    game_id=game_id,
                    bucket=bucket.value,
                    assigned_at=assigned_at,
                )
            )
        if not pending:
            return 0
        metadata = GameSplitAssignment.get_metadata()
        records = [row._to_db_record() for row in pending]
        result = self.db_client.insert(records, metadata, on_conflict="nothing")
        logger.info(
            "SplitRegistry ensure_games split_version=%s requested=%s new=%s skipped=%s",
            self.split_version,
            len(unique),
            result.rows_inserted,
            len(unique) - len(pending),
        )
        return int(result.rows_inserted)

    def fetch_buckets(self, game_ids: Sequence[str]) -> dict[str, SplitBucket]:
        """Load stored buckets for ``game_ids`` (missing ids omitted)."""
        unique = sorted({gid for gid in game_ids if gid})
        if not unique:
            return {}
        self.ensure_table()
        game_id_list = ", ".join(quote_literal(gid) for gid in unique)
        where = (
            f"{generate_ident_is_literal('split_version', self.split_version)} "
            f'AND "game_id" IN ({game_id_list})'
        )
        rows = GameSplitAssignment.fetch_all_from_db(self.db_client, where=where)
        return {row.game_id: SplitBucket(row.bucket) for row in rows}

    def split_datums(
        self,
        datums: list[TrainingDatum],
        *,
        assign_if_missing: bool = True,
        compute_disagree_frac: bool = True,
    ) -> GameSplitResult:
        """Partition datums using registry buckets (assign-on-read when enabled)."""
        if not datums:
            return split_datums_by_game(
                [],
                salt=self.split_version,
                compute_disagree_frac=compute_disagree_frac,
            )
        game_ids = sorted({d.game_id for d in datums})
        if assign_if_missing:
            self.ensure_games(game_ids)
        buckets = self.fetch_buckets(game_ids)
        missing = [gid for gid in game_ids if gid not in buckets]
        if missing:
            raise RuntimeError(
                f"Missing split assignments for {len(missing)} games "
                f"(split_version={self.split_version!r}); run backfill or ensure_games."
            )

        def bucket_for_game(game_id: str) -> SplitBucket:
            return buckets[game_id]

        return split_datums_by_game(
            datums,
            salt=self.split_version,
            compute_disagree_frac=compute_disagree_frac,
            bucket_for_game=bucket_for_game,
        )

    def backfill_eligible_games(self, *, batch_size: int = 500) -> BackfillResult:
        """Assign every training-eligible game not yet in the registry."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.ensure_table()
        self.db_client.ensure_tables(
            Move.get_metadata(),
            Game.get_metadata(),
            MoveCharacteristics.get_metadata(),
        )

        eligible = self._count_eligible_games()
        newly_assigned = 0
        already_assigned = 0
        offset = 0
        while offset < eligible:
            game_ids = self._fetch_eligible_game_ids(limit=batch_size, offset=offset)
            if not game_ids:
                break
            existing = self.fetch_buckets(game_ids)
            already_assigned += len(existing)
            newly_assigned += self.ensure_games(game_ids)
            offset += len(game_ids)
            logger.info(
                "Backfill progress split_version=%s offset=%s/%s batch=%s",
                self.split_version,
                offset,
                eligible,
                len(game_ids),
            )
        return BackfillResult(
            split_version=self.split_version,
            eligible_games=eligible,
            newly_assigned=newly_assigned,
            already_assigned=already_assigned,
        )

    def fetch_game_ids_for_bucket(self, bucket: SplitBucket) -> list[str]:
        """Return stored ``game_id``s for one bucket of this ``split_version``."""
        self.ensure_table()
        where = (
            f"{generate_ident_is_literal('split_version', self.split_version)} "
            f"AND {generate_ident_is_literal('bucket', bucket.value)}"
        )
        rows = GameSplitAssignment.fetch_all_from_db(
            self.db_client,
            where=where,
            order_by='"game_id" ASC',
        )
        return [row.game_id for row in rows]

    def exclude_holdout_games_sql(self, *, game_id_column: str = "g.game_id") -> str:
        """SQL fragment: true when ``game_id`` is not registry val/test (Phase 4 train filter)."""
        version_lit = quote_literal(self.split_version)
        return f"""NOT EXISTS (
            SELECT 1
            FROM ml.game_split_assignments gs
            WHERE gs.split_version = {version_lit}
              AND gs.game_id = {game_id_column}
              AND gs.bucket IN ('val', 'test')
        )"""

    def _where_game(self, game_id: str) -> str:
        return (
            f"{generate_ident_is_literal('split_version', self.split_version)} "
            f"AND {generate_ident_is_literal('game_id', game_id)}"
        )

    def _count_eligible_games(self) -> int:
        sql = f"SELECT COUNT(DISTINCT m.game_id) AS n{_ELIGIBLE_GAMES_SQL}"
        rows = self.db_client.engine.execute_parameterized_query(
            sql,
            {},
            session_settings=_MOVES_QUERY_SESSION_SETTINGS,
        )
        return int(rows[0]["n"]) if rows else 0

    def _fetch_eligible_game_ids(self, *, limit: int, offset: int) -> list[str]:
        sql = (
            f"SELECT DISTINCT m.game_id AS game_id{_ELIGIBLE_GAMES_SQL} "
            "ORDER BY m.game_id "
            "LIMIT :limit OFFSET :offset"
        )
        rows = self.db_client.engine.execute_parameterized_query(
            sql,
            {"limit": limit, "offset": offset},
            session_settings=_MOVES_QUERY_SESSION_SETTINGS,
        )
        return [str(row["game_id"]) for row in rows]


def get_split_registry(
    db_client: DatabaseClient | None = None,
    *,
    split_version: str = DEFAULT_SPLIT_SALT,
) -> SplitRegistry:
    return SplitRegistry(db_client or get_db_client(), split_version=split_version)
