from __future__ import annotations

import chess

from chess_teacher.pipelines.preprocessing.fen_characteristic import (
    DualSidedFenCharacteristicTransformation,
)
from chess_teacher.utils.chess_utils import fen_attack_pressure
from chess_teacher.utils.exception_utils import TransformationError


class AttackPressureTransformation(DualSidedFenCharacteristicTransformation):
    """Attack pressure on each color's pieces (attackers exceeding defenders)."""

    characteristic_name = "attack_pressure"

    def __init__(self) -> None:
        super().__init__(log_progress_percent=None)

    def evaluate_sides(self, fen: str, *, row: dict[str, object]) -> tuple[float, float]:
        del row
        try:
            board = chess.Board(fen)
        except ValueError as e:
            raise TransformationError(f"Invalid FEN for attack pressure: {fen!r}") from e
        return (
            fen_attack_pressure(board, chess.WHITE),
            fen_attack_pressure(board, chess.BLACK),
        )
