from __future__ import annotations

import chess

from chess_teacher.pipelines.preprocessing.fen_characteristic import (
    DualSidedFenCharacteristicTransformation,
)
from chess_teacher.utils.chess_utils import fen_pin_value
from chess_teacher.utils.exception_utils import TransformationError


class PinValueTransformation(DualSidedFenCharacteristicTransformation):
    """Pin burden on each color's pieces (pinning-piece-aware)."""

    characteristic_name = "pin_value"

    def __init__(self) -> None:
        super().__init__(log_progress_percent=None)

    def evaluate_sides(self, fen: str, *, row: dict[str, object]) -> tuple[float, float]:
        del row
        try:
            board = chess.Board(fen)
        except ValueError as e:
            raise TransformationError(f"Invalid FEN for pin value: {fen!r}") from e
        return fen_pin_value(board, chess.WHITE), fen_pin_value(board, chess.BLACK)
