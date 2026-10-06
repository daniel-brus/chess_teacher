from __future__ import annotations

import polars as pl

from chess_teacher.pipelines.ingestion.raw_games import RawGame
from chess_teacher.pipelines.modes import PipelineMode, preprocessing_transform_config
from chess_teacher.pipelines.preprocessing.games import Game
from chess_teacher.pipelines.preprocessing.move_characteristics import (
    AttackPressureTransformation,
    CandidateEvaluationsTransformation,
    DiagonalOpennessTransformation,
    HangingValueTransformation,
    KingSafetyTransformation,
    LegalMovesTransformation,
    MaterialBalanceTransformation,
    MeanRankTransformation,
    MoveContextTransformation,
    MoveFlagsTransformation,
    PawnTensionTransformation,
    PinValueTransformation,
    StockfishEvaluationTransformation,
    VerticalOpennessTransformation,
)
from chess_teacher.pipelines.preprocessing.move_extraction import ExtractUserMovesTransformation
from chess_teacher.pipelines.preprocessing.moves import Move, MoveCharacteristics
from chess_teacher.pipelines.preprocessing.opening_tables import RawEcoCode
from chess_teacher.pipelines.preprocessing.transformations import (
    ApplyChessComOpeningLookupTransformation,
    ApplyLichessOpeningNameTransformation,
    CleanPGNTransformation,
    DeriveOpeningTransformation,
    ExpandRawResponseTransformation,
    ExtractGameMetadataTransformation,
    ExtractPlayersAndResultTransformation,
    FilterGamesWithPGNTransformation,
)
from chess_teacher.platform.account import Account
from chess_teacher.utils.db.client import DatabaseClient, MergeStrategy
from chess_teacher.utils.exception_utils import PipelineError
from chess_teacher.utils.general_utils import (
    generate_ident_is_literal,
    quote_ident,
    quote_literal,
)
from chess_teacher.utils.pipeline_utils.pipeline_base import PipelineContext
from chess_teacher.utils.pipeline_utils.pipeline_steps import (
    LoadingStrategy,
    TransformStep,
)
from chess_teacher.utils.pipeline_utils.transformations import (
    CreateHashedIdTransformation,
    JoinWithTableTransformation,
    RenameColumnsTransformation,
)

# Cap peak RAM for fat raw_response / PGN frames on small VPS nodes.
_PREPROCESS_GAME_BATCH_SIZE = 500
# Move-level enrich: keyset pages via TransformStep.batch_size (parent already
# implements pagination). Keep well under node RAM when FENs + board metrics
# (or Stockfish MultiPV payloads) are held in-process.
_PREPROCESS_MOVE_BATCH_SIZE = 2000


class RawGamesToGamesStep(TransformStep):
    """Transform raw_games rows into enriched games rows for the current account."""

    def __init__(self, *, mode: PipelineMode = PipelineMode.INCREMENTAL) -> None:
        on, merge_strategy = preprocessing_transform_config(mode, incremental_on="game_id")
        super().__init__(
            name="RawGamesToGames",
            source_data_class=RawGame,
            target_data_class=Game,
            on=on,
            transformations=[
                ExpandRawResponseTransformation(),
                JoinWithTableTransformation(
                    with_data_class=Account,
                    columns=["account_id", "username", "platform"],
                ),
                FilterGamesWithPGNTransformation(),
                RenameColumnsTransformation({"pgn": "raw_pgn"}),
                ExtractGameMetadataTransformation(),
                ApplyLichessOpeningNameTransformation(),
                ExtractPlayersAndResultTransformation(),
                CleanPGNTransformation(),
                JoinWithTableTransformation(
                    with_data_class=RawEcoCode,
                    left_on=["eco_code"],
                    right_on=["eco_code"],
                    columns=["eco_code_id", "eco_code", "name", "pgn"],
                ),
                DeriveOpeningTransformation(),
                ApplyChessComOpeningLookupTransformation(),
            ],
            loading_strategy=LoadingStrategy.MERGE,
            merge_strategy=merge_strategy,
            batch_size=_PREPROCESS_GAME_BATCH_SIZE,
        )


class ExtractUserMovesStep(TransformStep):
    """Extract user moves from games into games.moves for the current account."""

    def __init__(self, *, mode: PipelineMode = PipelineMode.INCREMENTAL) -> None:
        on, merge_strategy = preprocessing_transform_config(mode, incremental_on="game_id")
        super().__init__(
            name="ExtractUserMoves",
            source_data_class=Game,
            target_data_class=Move,
            on=on,
            source_columns=["game_id", "account_id", "cleaned_pgn", "color", "variant"],
            transformations=[
                ExtractUserMovesTransformation(),
                CreateHashedIdTransformation(data_class=Move),
            ],
            loading_strategy=LoadingStrategy.MERGE,
            merge_strategy=merge_strategy,
            batch_size=_PREPROCESS_GAME_BATCH_SIZE,
        )


class EnrichCheapMoveCharacteristicsStep(TransformStep):
    """Board metrics + move flags into ``move_characteristics`` (no Stockfish).

    Inserts/updates rows so expensive Stockfish steps can checkpoint against
    existing ``move_id``s. Expensive columns stay NULL until the next step.
    """

    _SOURCE_COLUMNS: tuple[str, ...] = (
        "move_id",
        "game_id",
        "account_id",
        "fen_before",
        "fen_after",
        "move_uci",
        "previous_opponent_move_san",
        "previous_opponent_move_uci",
        "opponent_move_was_capture",
    )

    def __init__(self, *, mode: PipelineMode = PipelineMode.INCREMENTAL) -> None:
        on, merge_strategy = preprocessing_transform_config(mode, incremental_on="move_id")
        super().__init__(
            name="EnrichCheapMoveCharacteristics",
            source_data_class=Move,
            target_data_class=MoveCharacteristics,
            on=on,
            source_columns=list(self._SOURCE_COLUMNS),
            transformations=[
                MaterialBalanceTransformation(),
                VerticalOpennessTransformation(),
                DiagonalOpennessTransformation(),
                PawnTensionTransformation(),
                LegalMovesTransformation(),
                KingSafetyTransformation(),
                MeanRankTransformation(),
                AttackPressureTransformation(),
                HangingValueTransformation(),
                PinValueTransformation(),
                MoveContextTransformation(),
                MoveFlagsTransformation(),
            ],
            loading_strategy=LoadingStrategy.MERGE,
            merge_strategy=merge_strategy,
            batch_size=_PREPROCESS_MOVE_BATCH_SIZE,
        )

    def _load_records(
        self,
        db_client: DatabaseClient,
        context: PipelineContext,
        *,
        after_key: str | None = None,
    ) -> pl.DataFrame:
        """Skip the NOT EXISTS anti-join when account move/mc counts already match."""
        empty = pl.DataFrame({column: [] for column in self._SOURCE_COLUMNS})
        if self.on is not None and after_key is None and context.account_id is not None:
            moves_meta = Move.get_metadata()
            mc_meta = MoveCharacteristics.get_metadata()
            if db_client.table_exists(moves_meta) and db_client.table_exists(mc_meta):
                scope = generate_ident_is_literal("account_id", context.account_id)
                n_moves = db_client.get_row_count(moves_meta, where=scope)
                n_mc = db_client.get_row_count(mc_meta, where=scope)
                if n_moves == 0 or n_moves == n_mc:
                    self.logger.info(
                        f"[{self.name}] Fast-path skip: moves={n_moves} mc={n_mc} "
                        "(no missing cheap rows)."
                    )
                    return empty
        return super()._load_records(db_client, context, after_key=after_key)


class EnrichExpensiveMoveCharacteristicsStep(TransformStep):
    """Fill Stockfish evaluation + candidate_evaluations on incomplete mc rows.

    Incremental/retry load only rows missing expensive columns. Reprocess /
    full_reload recompute expensive columns for all rows in account scope.
    Never uses ``on=move_id`` (that would skip existing incomplete rows).
    Never uses full_sync (would risk deleting cheap-only columns).

    Custom join loader still honors ``TransformStep`` keyset batching via
    ``batch_size`` + ``after_key`` on ``move_id``.
    """

    _LOAD_COLUMNS: tuple[str, ...] = (
        "move_id",
        "game_id",
        "account_id",
        "fen_before",
        "fen_after",
        "move_uci",
        "ply",
    )

    def __init__(self, *, mode: PipelineMode = PipelineMode.INCREMENTAL) -> None:
        self._mode = mode
        # Always upsert: rows already exist from the cheap step; never full_sync.
        super().__init__(
            name="EnrichExpensiveMoveCharacteristics",
            source_data_class=Move,
            target_data_class=MoveCharacteristics,
            on=None,
            source_column="move_id",
            source_columns=list(self._LOAD_COLUMNS),
            transformations=[
                StockfishEvaluationTransformation(depth=12, log_progress_percent=5),
                CandidateEvaluationsTransformation(log_progress_percent=5),
            ],
            loading_strategy=LoadingStrategy.MERGE,
            merge_strategy=MergeStrategy.upsert(),
            batch_size=_PREPROCESS_MOVE_BATCH_SIZE,
        )

    def _load_records(
        self,
        db_client: DatabaseClient,
        context: PipelineContext,
        *,
        after_key: str | None = None,
    ) -> pl.DataFrame:
        """Load moves joined to incomplete (or all) move_characteristics rows."""
        moves_meta = Move.get_metadata()
        mc_meta = MoveCharacteristics.get_metadata()
        moves_sql = moves_meta.qualified_name_sql()
        mc_sql = mc_meta.qualified_name_sql()
        empty = pl.DataFrame({column: [] for column in self._LOAD_COLUMNS})

        if not db_client.table_exists(moves_meta) or not db_client.table_exists(mc_meta):
            self.logger.warning(f"[{self.name}] Source tables missing; using empty frame.")
            return empty

        if context.account_id is None:
            if context.user_id is not None:
                raise PipelineError(
                    f"[{self.name}] Cannot scope by user_id alone; account_id is required."
                )
        else:
            account_scope = generate_ident_is_literal("account_id", context.account_id)
            # Empty incomplete probe (partial index) before ordered join+limit.
            if (
                self._mode in (PipelineMode.INCREMENTAL, PipelineMode.RETRY)
                and after_key is None
                and not db_client.exists(
                    mc_meta,
                    f"{account_scope} AND {MoveCharacteristics.sql_expensive_incomplete('')}",
                )
            ):
                self.logger.info(
                    f"[{self.name}] Fast-path skip: no incomplete expensive rows "
                    f"for account_id={context.account_id}."
                )
                return empty

        context.progress_update(f"Reading characteristics rows from {mc_sql}...")
        select_cols = ", ".join(f"m.{quote_ident(col)}" for col in self._LOAD_COLUMNS)
        # Drive from mc so incomplete filters / partial index apply before the join.
        sql = (
            f"SELECT {select_cols}\n"
            f"FROM {mc_sql} AS mc\n"
            f"INNER JOIN {moves_sql} AS m ON m.{quote_ident('move_id')} = mc.{quote_ident('move_id')}"
        )
        clauses: list[str] = []
        if context.account_id is not None:
            clauses.append(f"mc.{generate_ident_is_literal('account_id', context.account_id)}")
        if self._mode in (PipelineMode.INCREMENTAL, PipelineMode.RETRY):
            clauses.append(MoveCharacteristics.sql_expensive_incomplete("mc"))
        if after_key is not None:
            clauses.append(f"mc.{quote_ident('move_id')} > {quote_literal(after_key)}")
        if clauses:
            sql += "\nWHERE " + " AND ".join(f"({c})" for c in clauses)
        sql += f"\nORDER BY mc.{quote_ident('move_id')}"
        if self.batch_size is not None:
            sql += f"\nLIMIT {int(self.batch_size)}"
        sql += ";"

        rows = db_client.engine.execute_parameterized_query(sql, {})
        self.logger.info(
            f"[{self.name}] Loaded {len(rows)} move row(s) for expensive enrichment "
            f"(mode={self._mode.value}, after_key={after_key!r})."
        )
        if not rows:
            return empty
        return pl.DataFrame(rows)
