from __future__ import annotations

import chess

from chess_teacher.pipelines.preprocessing.fen_characteristic import (
    DualSidedFenCharacteristicTransformation,
)
from chess_teacher.utils.chess_utils import fen_mean_rank
from chess_teacher.utils.exception_utils import TransformationError


class MeanRankTransformation(DualSidedFenCharacteristicTransformation):
    """Mean piece rank per color normalized 0-1 (higher = more advanced)."""

    characteristic_name = "mean_rank"

    def __init__(self) -> None:
        super().__init__(log_progress_percent=None)

    def evaluate_sides(self, fen: str, *, row: dict[str, object]) -> tuple[float, float]:
        del row
        try:
            board = chess.Board(fen)
        except ValueError as e:
            raise TransformationError(f"Invalid FEN for mean rank: {fen!r}") from e
        return fen_mean_rank(board, chess.WHITE), fen_mean_rank(board, chess.BLACK)
