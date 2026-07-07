from __future__ import annotations

import chess

from chess_teacher.pipelines.preprocessing.fen_characteristic import (
    DualSidedFenCharacteristicTransformation,
)
from chess_teacher.utils.chess_utils import fen_king_safety
from chess_teacher.utils.exception_utils import TransformationError


class KingSafetyTransformation(DualSidedFenCharacteristicTransformation):
    """King safety score per color (higher = safer)."""

    characteristic_name = "king_safety"

    def __init__(self) -> None:
        super().__init__(log_progress_percent=None)

    def evaluate_sides(self, fen: str, *, row: dict[str, object]) -> tuple[float, float]:
        del row
        try:
            board = chess.Board(fen)
        except ValueError as e:
            raise TransformationError(f"Invalid FEN for king safety: {fen!r}") from e
        return fen_king_safety(board, chess.WHITE), fen_king_safety(board, chess.BLACK)
