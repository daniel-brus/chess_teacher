from __future__ import annotations

from chess_teacher.pipelines.preprocessing.fen_characteristic import FenCharacteristicTransformation
from chess_teacher.utils.chess_utils import fen_vertical_openness
from chess_teacher.utils.exception_utils import TransformationError


class VerticalOpennessTransformation(FenCharacteristicTransformation):
    """Weighted file openness (0-8 scale)."""

    characteristic_name = "vertical_openness"

    def __init__(self) -> None:
        super().__init__(log_progress_percent=None)

    def evaluate(self, fen: str, *, row: dict[str, object]) -> float:
        del row
        try:
            return fen_vertical_openness(fen)
        except ValueError as e:
            raise TransformationError(f"Invalid FEN for vertical openness: {fen!r}") from e
