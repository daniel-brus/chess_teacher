from __future__ import annotations

import chess
import polars as pl

from chess_teacher.utils.chess_utils import (
    fen_game_phase,
    fen_has_castling_rights,
    fen_has_hanging_piece,
    fen_is_in_check,
)
from chess_teacher.utils.exception_utils import TransformationError
from chess_teacher.utils.pipeline_utils.dataframe_transformation import DataFrameTransformation


class MoveContextTransformation(DataFrameTransformation):
    """Row-wise position context from ``fen_before`` (side to move is always the user)."""

    _REQUIRED_COLUMNS = ("fen_before",)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        missing = [column for column in self._REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise TransformationError(f"MoveContextTransformation requires columns {missing}.")
        if df.height == 0:
            return df

        is_in_check_before: list[bool] = []
        user_has_hanging_piece_before: list[bool] = []
        opponent_has_hanging_piece_before: list[bool] = []
        has_castling_rights_before: list[bool] = []
        is_opening: list[bool] = []
        is_middle_game: list[bool] = []
        is_end_game: list[bool] = []

        for fen_before in df["fen_before"].cast(pl.Utf8).to_list():
            try:
                board = chess.Board(str(fen_before))
            except ValueError as e:
                raise TransformationError(f"Invalid FEN for move context: {fen_before!r}") from e

            user_color = board.turn
            opponent_color = not user_color
            opening, middle_game, end_game = fen_game_phase(str(fen_before))

            is_in_check_before.append(fen_is_in_check(str(fen_before)))
            user_has_hanging_piece_before.append(fen_has_hanging_piece(board, user_color))
            opponent_has_hanging_piece_before.append(fen_has_hanging_piece(board, opponent_color))
            has_castling_rights_before.append(fen_has_castling_rights(board, user_color))
            is_opening.append(opening)
            is_middle_game.append(middle_game)
            is_end_game.append(end_game)

        return df.with_columns(
            pl.Series("is_in_check_before", is_in_check_before, dtype=pl.Boolean),
            pl.Series(
                "user_has_hanging_piece_before",
                user_has_hanging_piece_before,
                dtype=pl.Boolean,
            ),
            pl.Series(
                "opponent_has_hanging_piece_before",
                opponent_has_hanging_piece_before,
                dtype=pl.Boolean,
            ),
            pl.Series("has_castling_rights_before", has_castling_rights_before, dtype=pl.Boolean),
            pl.Series("is_opening", is_opening, dtype=pl.Boolean),
            pl.Series("is_middle_game", is_middle_game, dtype=pl.Boolean),
            pl.Series("is_end_game", is_end_game, dtype=pl.Boolean),
        )
