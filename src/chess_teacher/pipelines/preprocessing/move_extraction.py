from __future__ import annotations

import re
from io import StringIO
from typing import Any

import chess
import chess.pgn
import polars as pl

from chess_teacher.utils.chess_utils import Color
from chess_teacher.utils.exception_utils import TransformationError
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.pipeline_utils.transformations import DataFrameTransformation

logger = get_logger()

_RESULT_SUFFIX_RE = re.compile(r"\s*(?:1-0|0-1|1/2-1/2|\*)\s*$")
_BLACK_MOVE_NUMBER_RE = re.compile(r"\b\d+\.{2,3}\s*")
_WHITE_MOVE_NUMBER_RE = re.compile(r"\b\d+\.\s")

_MOVE_STRUCT = pl.Struct({
    "game_id": pl.Utf8,
    "account_id": pl.Utf8,
    "move_nr": pl.Int64,
    "ply": pl.Int64,
    "move_san": pl.Utf8,
    "move_uci": pl.Utf8,
    "fen_before": pl.Utf8,
    "fen_after": pl.Utf8,
    "previous_opponent_move_san": pl.Utf8,
    "previous_opponent_move_uci": pl.Utf8,
    "opponent_move_was_capture": pl.Boolean,
})

_MOVE_OUTPUT_SCHEMA = pl.Schema({
    "game_id": pl.Utf8,
    "account_id": pl.Utf8,
    "move_nr": pl.Int64,
    "ply": pl.Int64,
    "move_san": pl.Utf8,
    "move_uci": pl.Utf8,
    "fen_before": pl.Utf8,
    "fen_after": pl.Utf8,
    "previous_opponent_move_san": pl.Utf8,
    "previous_opponent_move_uci": pl.Utf8,
    "opponent_move_was_capture": pl.Boolean,
})

_GAME_INPUT_COLUMNS = ("game_id", "account_id", "cleaned_pgn", "color", "variant")


def tokenize_cleaned_movetext(cleaned_pgn: str) -> list[str]:
    """Split cleaned movetext into SAN tokens (no PGN parser)."""
    body = cleaned_pgn.strip()
    body = _RESULT_SUFFIX_RE.sub("", body).strip()
    body = _BLACK_MOVE_NUMBER_RE.sub("", body)
    body = _WHITE_MOVE_NUMBER_RE.sub("", body)
    body = re.sub(r"\s+", " ", body).strip()
    if not body:
        return []
    return body.split(" ")


def _rows_from_sans(
    *,
    game_id: str,
    sans: list[str],
    user_turn: chess.Color,
) -> list[dict[str, object]]:
    board = chess.Board()
    ply = 0
    move_nr = 0
    rows: list[dict[str, object]] = []
    previous_opponent_move_san: str | None = None
    previous_opponent_move_uci: str | None = None
    opponent_move_was_capture = False

    for san in sans:
        ply += 1
        if board.turn == user_turn:
            move_nr += 1
            fen_before = board.fen()
            move = board.push_san(san)
            rows.append({
                "game_id": game_id,
                "move_nr": move_nr,
                "ply": ply,
                "move_san": san,
                "move_uci": move.uci(),
                "fen_before": fen_before,
                "fen_after": board.fen(),
                "previous_opponent_move_san": previous_opponent_move_san,
                "previous_opponent_move_uci": previous_opponent_move_uci,
                "opponent_move_was_capture": opponent_move_was_capture,
            })
        else:
            move = board.parse_san(san)
            opponent_move_was_capture = board.is_capture(move)
            previous_opponent_move_san = san
            previous_opponent_move_uci = move.uci()
            board.push_san(san)

    return rows


def _rows_from_pgn_parser(
    *,
    game_id: str,
    cleaned_pgn: str,
    user_turn: chess.Color,
) -> list[dict[str, object]]:
    try:
        game = chess.pgn.read_game(StringIO(f'[Event "?"]\n\n{cleaned_pgn}'))
    except (ValueError, chess.InvalidMoveError):
        return []
    if game is None:
        return []

    board = game.board()
    node: chess.pgn.Game | chess.pgn.ChildNode = game
    ply = 0
    move_nr = 0
    rows: list[dict[str, object]] = []
    previous_opponent_move_san: str | None = None
    previous_opponent_move_uci: str | None = None
    opponent_move_was_capture = False

    while node.variations:
        ply += 1
        next_node = node.variation(0)
        move = next_node.move
        if move is None:
            break
        if board.turn == user_turn:
            move_nr += 1
            fen_before = board.fen()
            move_san = board.san(move)
            board.push(move)
            rows.append({
                "game_id": game_id,
                "move_nr": move_nr,
                "ply": ply,
                "move_san": move_san,
                "move_uci": move.uci(),
                "fen_before": fen_before,
                "fen_after": board.fen(),
                "previous_opponent_move_san": previous_opponent_move_san,
                "previous_opponent_move_uci": previous_opponent_move_uci,
                "opponent_move_was_capture": opponent_move_was_capture,
            })
        else:
            opponent_move_was_capture = board.is_capture(move)
            previous_opponent_move_san = board.san(move)
            previous_opponent_move_uci = move.uci()
            board.push(move)
        node = next_node

    return rows


def extract_user_moves(
    *,
    game_id: str,
    cleaned_pgn: str,
    color: Color,
    variant: str = "standard",
) -> list[dict[str, object]]:
    """Extract one row per user move from cleaned movetext."""
    if variant != "standard":
        return []
    if not cleaned_pgn or not cleaned_pgn.strip():
        return []

    user_turn = chess.WHITE if color == Color.WHITE else chess.BLACK
    sans = tokenize_cleaned_movetext(cleaned_pgn)
    if sans:
        try:
            return _rows_from_sans(game_id=game_id, sans=sans, user_turn=user_turn)
        except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError):
            pass

    return _rows_from_pgn_parser(
        game_id=game_id,
        cleaned_pgn=cleaned_pgn,
        user_turn=user_turn,
    )


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
            .explode("_moves", empty_as_null=True)
            .unnest("_moves")
        )
        if result.height == 0:
            return pl.DataFrame(schema=_MOVE_OUTPUT_SCHEMA)

        return result.select(_MOVE_OUTPUT_SCHEMA.names()).cast(_MOVE_OUTPUT_SCHEMA)
