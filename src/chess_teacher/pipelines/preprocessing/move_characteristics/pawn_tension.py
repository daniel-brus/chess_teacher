from __future__ import annotations

from chess_teacher.pipelines.preprocessing.fen_characteristic import FenCharacteristicTransformation
from chess_teacher.utils.chess_utils import fen_pawn_tension
from chess_teacher.utils.exception_utils import TransformationError


class PawnTensionTransformation(FenCharacteristicTransformation):
    """Count of pawn pairs on adjacent files with mutual capture potential."""

    characteristic_name = "pawn_tension"

    def __init__(self) -> None:
        super().__init__(log_progress_percent=None)

    def evaluate(self, fen: str, *, row: dict[str, object]) -> float:
        del row
        try:
            return fen_pawn_tension(fen)
        except ValueError as e:
            raise TransformationError(f"Invalid FEN for pawn tension: {fen!r}") from e
