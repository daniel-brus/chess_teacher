from __future__ import annotations

from typing import Any

import polars as pl

from chess_teacher.ingestion.move_extraction_core import (
    extract_user_moves,  # re-exported for tests
)
from chess_teacher.ingestion.moves import Move
from chess_teacher.pipelines.transformations import DataFrameTransformation
from chess_teacher.utils.chess_utils import Color
from chess_teacher.utils.db.client import DatabaseClient, get_db_client
from chess_teacher.utils.exception_utils import TransformationError
from chess_teacher.utils.general_utils import generate_ident_is_literal, quote_ident
from chess_teacher.utils.logging import get_logger

logger = get_logger()

_MOVE_STRUCT = pl.Struct({
    "game_id": pl.Utf8,
    "account_id": pl.Utf8,
    "move_nr": pl.Int64,
    "ply": pl.Int64,
    "move_san": pl.Utf8,
    "move_uci": pl.Utf8,
    "fen_before": pl.Utf8,
    "fen_after": pl.Utf8,
})

_MOVE_OUTPUT_SCHEMA = {
    "game_id": pl.Utf8,
    "account_id": pl.Utf8,
    "move_nr": pl.Int64,
    "ply": pl.Int64,
    "move_san": pl.Utf8,
    "move_uci": pl.Utf8,
    "fen_before": pl.Utf8,
    "fen_after": pl.Utf8,
}

_GAME_INPUT_COLUMNS = ("game_id", "account_id", "cleaned_pgn", "color", "variant")


def _moves_for_game_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Return move rows for one game (empty list when there are no user moves)."""
    extracted = extract_user_moves(
        game_id=str(row["game_id"]),
        cleaned_pgn=str(row["cleaned_pgn"]),
        color=Color(str(row["color"])),
        variant=str(row.get("variant") or "standard"),
    )
    account_id = row["account_id"]
    for move_row in extracted:
        move_row["account_id"] = account_id
    return extracted


class FilterGamesAlreadyInMovesTransformation(DataFrameTransformation):
    """Drop games that already have rows in ``games.moves`` for the same account."""

    REQUIRED_COLUMNS = ("game_id", "account_id")

    def __init__(
        self,
        *,
        moves_data_class: type[Move] = Move,
        db_client: DatabaseClient | None = None,
    ) -> None:
        self.moves_metadata = moves_data_class.get_metadata()
        self.db_client = db_client or get_db_client()

    def _existing_game_ids(self, account_id: str) -> set[str]:
        if not self.db_client.table_exists(self.moves_metadata):
            return set()

        where = generate_ident_is_literal("account_id", account_id)
        sql = (
            f"SELECT DISTINCT {quote_ident('game_id')} "
            f"FROM {self.moves_metadata.qualified_name_sql()} "
            f"WHERE {where};"
        )
        rows = self.db_client.engine.execute_parameterized_query(sql, {})
        return {str(row["game_id"]) for row in rows}

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        for column in self.REQUIRED_COLUMNS:
            if column not in df.columns:
                logger.log_and_raise(
                    TransformationError(
                        f"Column {column!r} is required to filter games already in moves."
                    )
                )

        if df.height == 0:
            return df

        account_ids = df["account_id"].unique().to_list()
        if len(account_ids) != 1:
            logger.log_and_raise(
                TransformationError(
                    "FilterGamesAlreadyInMovesTransformation expects a single account_id "
                    f"per batch, got {len(account_ids)}: {account_ids}"
                )
            )

        existing_game_ids = self._existing_game_ids(str(account_ids[0]))
        if not existing_game_ids:
            return df

        before = df.height
        result = df.filter(~pl.col("game_id").is_in(existing_game_ids))
        skipped = before - result.height
        if skipped:
            logger.info(
                "FilterGamesAlreadyInMovesTransformation skipped %d game(s) "
                "already present in %s (%d -> %d).",
                skipped,
                self.moves_metadata.qualified_name_sql(),
                before,
                result.height,
            )
        return result


class ExtractUserMovesTransformation(DataFrameTransformation):
    """Expand games into one row per user move."""

    REQUIRED_COLUMNS = ("game_id", "account_id", "cleaned_pgn", "color")

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        for column in self.REQUIRED_COLUMNS:
            if column not in df.columns:
                logger.log_and_raise(
                    TransformationError(f"Column {column!r} is required to extract user moves.")
                )

        if df.height == 0:
            return pl.DataFrame(schema=_MOVE_OUTPUT_SCHEMA)

        if "variant" in df.columns:
            working = df.with_columns(pl.col("variant").fill_null("standard"))
        else:
            working = df.with_columns(pl.lit("standard").alias("variant"))

        skipped_variant = working.filter(pl.col("variant") != "standard").height
        standard = working.filter(pl.col("variant") == "standard")

        if standard.height == 0:
            if skipped_variant:
                logger.warning(
                    "ExtractUserMovesTransformation skipped %d game(s) with non-standard variant.",
                    skipped_variant,
                )
            return pl.DataFrame(schema=_MOVE_OUTPUT_SCHEMA)

        try:
            expanded = standard.with_columns(
                pl
                .struct(list(_GAME_INPUT_COLUMNS))
                .map_elements(_moves_for_game_row, return_dtype=pl.List(_MOVE_STRUCT))
                .alias("_moves")
            )
        except Exception as e:
            logger.log_and_raise(TransformationError(f"Failed to extract user moves: {e}"))

        skipped_unparseable = expanded.filter(
            pl.col("_moves").list.len() == 0,
            pl.col("cleaned_pgn").str.strip_chars() != "",
        ).height

        if skipped_variant:
            logger.warning(
                "ExtractUserMovesTransformation skipped %d game(s) with non-standard variant.",
                skipped_variant,
            )
        if skipped_unparseable:
            logger.warning(
                "ExtractUserMovesTransformation could not parse %d game(s) with movetext.",
                skipped_unparseable,
            )

        result = (
            expanded
            .filter(pl.col("_moves").list.len() > 0)
            .select("_moves")
            .explode("_moves")
            .unnest("_moves")
        )
        if result.height == 0:
            return pl.DataFrame(schema=_MOVE_OUTPUT_SCHEMA)

        return result.select(list(_MOVE_OUTPUT_SCHEMA.keys())).cast(_MOVE_OUTPUT_SCHEMA)
