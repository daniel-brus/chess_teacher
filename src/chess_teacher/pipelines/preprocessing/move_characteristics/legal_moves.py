from __future__ import annotations

from chess_teacher.pipelines.preprocessing.fen_characteristic import (
    DualSidedFenCharacteristicTransformation,
)
from chess_teacher.utils.chess_utils import fen_legal_moves
from chess_teacher.utils.exception_utils import TransformationError


class LegalMovesTransformation(DualSidedFenCharacteristicTransformation):
    """Legal move counts for white and black (side-to-move ignored)."""

    characteristic_name = "legal_moves"

    def __init__(self) -> None:
        super().__init__(log_progress_percent=None)

    def evaluate_sides(self, fen: str, *, row: dict[str, object]) -> tuple[float, float]:
        del row
        try:
            return fen_legal_moves(fen)
        except ValueError as e:
            raise TransformationError(f"Invalid FEN for legal moves: {fen!r}") from e
