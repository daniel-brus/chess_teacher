from __future__ import annotations

from chess_teacher.pipelines.preprocessing.fen_characteristic import FenCharacteristicTransformation
from chess_teacher.utils.chess_utils import material_balance_white_pov
from chess_teacher.utils.exception_utils import TransformationError


class MaterialBalanceTransformation(FenCharacteristicTransformation):
    """Material balance (white minus black) from white's perspective."""

    characteristic_name = "material_balance"

    def __init__(self) -> None:
        super().__init__(log_progress_percent=None)

    def evaluate(self, fen: str, *, row: dict[str, object]) -> float:
        del row
        try:
            return material_balance_white_pov(fen)
        except ValueError as e:
            raise TransformationError(f"Invalid FEN for material balance: {fen!r}") from e
