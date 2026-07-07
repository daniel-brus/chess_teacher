from __future__ import annotations

import chess
import polars as pl

from chess_teacher.utils.chess_utils import (
    move_created_fork,
    move_gave_check,
    move_is_capture,
    move_is_castle,
    move_is_en_passant,
    move_is_promotion,
)
from chess_teacher.utils.exception_utils import TransformationError
from chess_teacher.utils.pipeline_utils.dataframe_transformation import DataFrameTransformation


class MoveFlagsTransformation(DataFrameTransformation):
    """Row-wise move semantics: capture, castle, check, and fork flags."""

    _REQUIRED_COLUMNS = ("fen_before", "fen_after", "move_uci", "opponent_move_was_capture")

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        missing = [column for column in self._REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise TransformationError(f"MoveFlagsTransformation requires columns {missing}.")
        if df.height == 0:
            return df

        is_capture: list[bool] = []
        is_castle: list[bool] = []
        gave_check: list[bool] = []
        created_fork: list[bool] = []
        is_promotion: list[bool] = []
        is_en_passant: list[bool] = []
        is_recapture: list[bool] = []

        for fen_before, fen_after, move_uci, opponent_move_was_capture in zip(
            df["fen_before"].cast(pl.Utf8).to_list(),
            df["fen_after"].cast(pl.Utf8).to_list(),
            df["move_uci"].cast(pl.Utf8).to_list(),
            df["opponent_move_was_capture"].cast(pl.Boolean).to_list(),
            strict=True,
        ):
            try:
                board = chess.Board(str(fen_before))
                move = chess.Move.from_uci(str(move_uci))
            except ValueError as e:
                raise TransformationError(
                    f"Invalid move data: fen_before={fen_before!r}, move_uci={move_uci!r}"
                ) from e

            if move not in board.legal_moves:
                raise TransformationError(f"Illegal move {move_uci!r} for position {fen_before!r}")

            capture = move_is_capture(board, move)
            is_capture.append(capture)
            is_castle.append(move_is_castle(move))
            gave_check.append(move_gave_check(board, move))
            created_fork.append(move_created_fork(str(fen_before), str(fen_after), str(move_uci)))
            is_promotion.append(move_is_promotion(move))
            is_en_passant.append(move_is_en_passant(board, move))
            is_recapture.append(capture and bool(opponent_move_was_capture))

        return df.with_columns(
            pl.Series("is_capture", is_capture, dtype=pl.Boolean),
            pl.Series("is_castle", is_castle, dtype=pl.Boolean),
            pl.Series("gave_check", gave_check, dtype=pl.Boolean),
            pl.Series("created_fork", created_fork, dtype=pl.Boolean),
            pl.Series("is_promotion", is_promotion, dtype=pl.Boolean),
            pl.Series("is_en_passant", is_en_passant, dtype=pl.Boolean),
            pl.Series("is_recapture", is_recapture, dtype=pl.Boolean),
        )
