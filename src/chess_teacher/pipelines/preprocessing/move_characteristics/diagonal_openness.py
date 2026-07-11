from __future__ import annotations

from chess_teacher.pipelines.preprocessing.fen_characteristic import FenCharacteristicTransformation
from chess_teacher.utils.chess_utils import fen_diagonal_openness
from chess_teacher.utils.exception_utils import TransformationError


class DiagonalOpennessTransformation(FenCharacteristicTransformation):
    """Weighted openness for six center diagonals (0-6 scale)."""

    characteristic_name = "diagonal_openness"

    def __init__(self) -> None:
        super().__init__(log_progress_percent=None)

    def evaluate(self, fen: str, *, row: dict[str, object]) -> float:
        del row
        try:
            return fen_diagonal_openness(fen)
        except ValueError as e:
            raise TransformationError(f"Invalid FEN for diagonal openness: {fen!r}") from e
