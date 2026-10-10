"""Persistent game split assignments (Phase 1b).

Stores ``(split_version, game_id) → bucket`` in ``ml.game_split_assignments``.
Assignment rule matches ``splits.game_split_bucket`` — registry is persistence only.

See ``.agents/docs/ml-training-roadmap.md`` Phase 1b.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

from chess_teacher.pipelines.neural_network.create_training_set import TrainingDatum
from chess_teacher.pipelines.neural_network.models import (
    ATTEMPT_COLUMN_PERSONAL,
    PROCESSED_FLAG_BASELINE,
    PROCESSED_FLAG_PERSONAL,
    GameSplitAssignment,
    attempt_column_for_flag,
    require_processed_flag,
)
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
from chess_teacher.utils.general_utils import (
    generate_ident_is_literal,
    get_current_datetime,
    quote_ident,
    quote_literal,
)
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
                    already_processed_baseline=None,
                    already_processed_personal=None,
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

    def backfill_eligible_games(
        self,
        *,
        batch_size: int = 500,
        account_id: str | None = None,
    ) -> BackfillResult:
        """Assign training-eligible games not yet in the registry.

        When ``account_id`` is set (daily user pipeline), load all eligible
        ``game_id``s for that account in one query, then ``ensure_games`` in
        chunks. Platform-wide repair (no ``account_id``) keeps OFFSET batches
        so a huge catalog is not held in memory at once.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.ensure_table()
        self.db_client.ensure_tables(
            Move.get_metadata(),
            Game.get_metadata(),
            MoveCharacteristics.get_metadata(),
        )
        if account_id is not None:
            return self._backfill_eligible_games_for_account(
                account_id,
                batch_size=batch_size,
            )
        return self._backfill_eligible_games_platform_wide(batch_size=batch_size)

    def _backfill_eligible_games_for_account(
        self,
        account_id: str,
        *,
        batch_size: int,
    ) -> BackfillResult:
        """One-shot eligible-id list for one account, then chunked inserts."""
        if not account_id:
            raise ValueError("account_id is required")
        eligible = self._count_eligible_games(account_id=account_id)
        game_ids = self._fetch_all_eligible_game_ids(account_id=account_id)
        logger.info(
            "Account split backfill loaded ids split_version=%s account_id=%s "
            "eligible=%s ids=%s chunk=%s",
            self.split_version,
            account_id,
            eligible,
            len(game_ids),
            batch_size,
        )
        newly_assigned = 0
        already_assigned = 0
        total = len(game_ids)
        for offset in range(0, total, batch_size):
            chunk = game_ids[offset : offset + batch_size]
            existing = self.fetch_buckets(chunk)
            already_assigned += len(existing)
            newly_assigned += self.ensure_games(chunk)
            logger.info(
                "Backfill progress split_version=%s account_id=%s offset=%s/%s batch=%s",
                self.split_version,
                account_id,
                offset + len(chunk),
                total,
                len(chunk),
            )
        return BackfillResult(
            split_version=self.split_version,
            eligible_games=eligible,
            newly_assigned=newly_assigned,
            already_assigned=already_assigned,
        )

    def _backfill_eligible_games_platform_wide(self, *, batch_size: int) -> BackfillResult:
        """OFFSET-batched scan for platform-wide repair tools."""
        eligible = self._count_eligible_games(account_id=None)
        newly_assigned = 0
        already_assigned = 0
        offset = 0
        while offset < eligible:
            game_ids = self._fetch_eligible_game_ids(
                limit=batch_size,
                offset=offset,
                account_id=None,
            )
            if not game_ids:
                break
            existing = self.fetch_buckets(game_ids)
            already_assigned += len(existing)
            newly_assigned += self.ensure_games(game_ids)
            offset += len(game_ids)
            logger.info(
                "Backfill progress split_version=%s account_id=%s offset=%s/%s batch=%s",
                self.split_version,
                None,
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

    def ensure_eligible_games_for_account(
        self,
        account_id: str,
        *,
        batch_size: int = 2000,
    ) -> BackfillResult:
        """Assign eligible games for one account (idempotent).

        Default chunk size is for ``ensure_games`` inserts only; eligible ids
        are loaded once per account (not OFFSET-scanned).
        """
        if not account_id:
            raise ValueError("account_id is required")
        return self.backfill_eligible_games(batch_size=batch_size, account_id=account_id)

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

    def mark_processed(
        self,
        game_ids: Sequence[str],
        *,
        flag_column: str = PROCESSED_FLAG_BASELINE,
        processed_at: datetime | None = None,
    ) -> int:
        """Set the processed flag on **train** rows only. Returns rows updated.

        Never writes val/test. No-op when ``game_ids`` is empty. Caller must
        invoke this only after a successful fit.
        """
        unique = sorted({gid for gid in game_ids if gid})
        if not unique:
            return 0
        flag = require_processed_flag(flag_column)
        self.ensure_table()
        when = processed_at or get_current_datetime()
        metadata = GameSplitAssignment.get_metadata()
        updated = 0
        batch_size = 500
        for offset in range(0, len(unique), batch_size):
            chunk = unique[offset : offset + batch_size]
            game_id_list = ", ".join(quote_literal(gid) for gid in chunk)
            where = (
                f"{generate_ident_is_literal('split_version', self.split_version)} "
                f"AND {generate_ident_is_literal('bucket', SplitBucket.TRAIN.value)} "
                f"AND {quote_ident(flag)} IS NULL "
                f'AND "game_id" IN ({game_id_list})'
            )
            updated += self.db_client.update_where(
                metadata,
                {flag: when},
                where,
            )
        logger.info(
            "SplitRegistry mark_processed split_version=%s flag=%s requested=%s updated=%s",
            self.split_version,
            flag,
            len(unique),
            updated,
        )
        return updated

    def note_attempt(
        self,
        game_ids: Sequence[str],
        *,
        flag_column: str = PROCESSED_FLAG_BASELINE,
    ) -> int:
        """Count one missed promotion on train rows that are still unmarked.

        Each miss sorts the games further back. They stay eligible until a
        later round promotes them. Val and test rows are never updated.
        """
        unique = sorted({gid for gid in game_ids if gid})
        if not unique:
            return 0
        flag = require_processed_flag(flag_column)
        attempt = attempt_column_for_flag(flag)
        self.ensure_table()
        metadata = GameSplitAssignment.get_metadata()
        updated = 0
        batch_size = 500
        for offset in range(0, len(unique), batch_size):
            chunk = unique[offset : offset + batch_size]
            game_id_list = ", ".join(quote_literal(gid) for gid in chunk)
            where = (
                f"{generate_ident_is_literal('split_version', self.split_version)} "
                f"AND {generate_ident_is_literal('bucket', SplitBucket.TRAIN.value)} "
                f"AND {quote_ident(flag)} IS NULL "
                f'AND "game_id" IN ({game_id_list})'
            )
            sql = (
                f"UPDATE {metadata.qualified_name_sql()} "
                f"SET {quote_ident(attempt)} = COALESCE({quote_ident(attempt)}, 0) + 1 "
                f"WHERE {where}"
            )
            updated += self.db_client.engine.execute_write(sql, {})
        logger.info(
            "SplitRegistry note_attempt split_version=%s flag=%s requested=%s updated=%s",
            self.split_version,
            flag,
            len(unique),
            updated,
        )
        return updated

    def clear_personal_queue_for_accounts(self, account_ids: Sequence[str]) -> int:
        """Drop personal marks and attempt counts for one user's games.

        Used once when that user starts a new parent-baseline lineage.
        """
        unique = sorted({aid for aid in account_ids if aid})
        if not unique:
            return 0
        self.ensure_table()
        ids_sql = ", ".join(quote_literal(aid) for aid in unique)
        where = (
            f"{generate_ident_is_literal('split_version', self.split_version)} "
            f'AND "game_id" IN (SELECT "game_id" FROM games.games '
            f'WHERE "account_id" IN ({ids_sql}))'
        )
        updated = self.db_client.update_where(
            GameSplitAssignment.get_metadata(),
            {
                PROCESSED_FLAG_PERSONAL: None,
                ATTEMPT_COLUMN_PERSONAL: 0,
            },
            where,
        )
        logger.info(
            "SplitRegistry clear_personal_queue split_version=%s accounts=%s updated=%s",
            self.split_version,
            len(unique),
            updated,
        )
        return updated

    def clear_processed(
        self,
        *,
        flag_column: str = PROCESSED_FLAG_BASELINE,
        game_ids: Sequence[str] | None = None,
    ) -> int:
        """NULL the processed flag (fresh experiment / personal reset)."""
        flag = require_processed_flag(flag_column)
        self.ensure_table()
        metadata = GameSplitAssignment.get_metadata()
        where = generate_ident_is_literal("split_version", self.split_version)
        unique: list[str] = []
        if game_ids is not None:
            unique = sorted({gid for gid in game_ids if gid})
            if not unique:
                return 0
            game_id_list = ", ".join(quote_literal(gid) for gid in unique)
            where = f'{where} AND "game_id" IN ({game_id_list})'
        updated = self.db_client.update_where(metadata, {flag: None}, where)
        logger.info(
            "SplitRegistry clear_processed split_version=%s flag=%s scoped=%s updated=%s",
            self.split_version,
            flag,
            len(unique) if game_ids is not None else "all",
            updated,
        )
        return updated

    def reset_training_queue_markers(self, *, dry_run: bool = False) -> int:
        """Reset baseline and personal queue state for train-bucket rows only.

        Validation/test assignments, model rows, and training cutoffs are left
        unchanged. Returns the number of rows that would be or were reset.
        """
        self.ensure_table()
        metadata = GameSplitAssignment.get_metadata()
        where = (
            f"{generate_ident_is_literal('split_version', self.split_version)} "
            f"AND {generate_ident_is_literal('bucket', SplitBucket.TRAIN.value)} "
            'AND ("already_processed_baseline" IS NOT NULL '
            'OR "already_processed_personal" IS NOT NULL '
            'OR "baseline_train_attempts" <> 0 '
            'OR "personal_train_attempts" <> 0)'
        )
        matched = self.db_client.get_row_count(metadata, where=where)
        if dry_run or matched == 0:
            logger.info(
                "SplitRegistry reset_training_queue_markers split_version=%s dry_run=%s matched=%s",
                self.split_version,
                dry_run,
                matched,
            )
            return matched

        updated = self.db_client.update_where(
            metadata,
            {
                PROCESSED_FLAG_BASELINE: None,
                PROCESSED_FLAG_PERSONAL: None,
                "baseline_train_attempts": 0,
                ATTEMPT_COLUMN_PERSONAL: 0,
            },
            where,
        )
        logger.info(
            "SplitRegistry reset_training_queue_markers split_version=%s matched=%s updated=%s",
            self.split_version,
            matched,
            updated,
        )
        return updated

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

    def _eligible_from_sql(self, *, account_id: str | None = None) -> tuple[str, dict[str, object]]:
        sql = _ELIGIBLE_GAMES_SQL
        params: dict[str, object] = {}
        if account_id is not None:
            sql += " AND g.account_id = :account_id"
            params["account_id"] = account_id
        return sql, params

    def _count_eligible_games(self, *, account_id: str | None = None) -> int:
        from_sql, params = self._eligible_from_sql(account_id=account_id)
        sql = f"SELECT COUNT(DISTINCT m.game_id) AS n{from_sql}"
        rows = self.db_client.engine.execute_parameterized_query(
            sql,
            params,
            session_settings=_MOVES_QUERY_SESSION_SETTINGS,
        )
        return int(rows[0]["n"]) if rows else 0

    def _fetch_eligible_game_ids(
        self,
        *,
        limit: int,
        offset: int,
        account_id: str | None = None,
    ) -> list[str]:
        from_sql, params = self._eligible_from_sql(account_id=account_id)
        sql = (
            f"SELECT DISTINCT m.game_id AS game_id{from_sql} "
            "ORDER BY m.game_id "
            "LIMIT :limit OFFSET :offset"
        )
        params = {**params, "limit": limit, "offset": offset}
        rows = self.db_client.engine.execute_parameterized_query(
            sql,
            params,
            session_settings=_MOVES_QUERY_SESSION_SETTINGS,
        )
        return [str(row["game_id"]) for row in rows]

    def _fetch_all_eligible_game_ids(self, *, account_id: str) -> list[str]:
        """All eligible ``game_id``s for one account (no OFFSET). Account-scoped only."""
        if not account_id:
            raise ValueError("account_id is required for one-shot eligible fetch")
        from_sql, params = self._eligible_from_sql(account_id=account_id)
        sql = f"SELECT DISTINCT m.game_id AS game_id{from_sql} ORDER BY m.game_id"
        rows = self.db_client.engine.execute_parameterized_query(
            sql,
            params,
            session_settings=_MOVES_QUERY_SESSION_SETTINGS,
        )
        return [str(row["game_id"]) for row in rows]


def get_split_registry(
    db_client: DatabaseClient | None = None,
    *,
    split_version: str = DEFAULT_SPLIT_SALT,
) -> SplitRegistry:
    return SplitRegistry(db_client or get_db_client(), split_version=split_version)


def clear_personal_processed(
    db_client: DatabaseClient,
    game_ids: Sequence[str],
    *,
    split_version: str = DEFAULT_SPLIT_SALT,
) -> int:
    """NULL personal flags for one user's ``game_id``s (tests + later promote hook)."""
    return get_split_registry(db_client, split_version=split_version).clear_processed(
        flag_column=PROCESSED_FLAG_PERSONAL,
        game_ids=game_ids,
    )
