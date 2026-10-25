"""Policy indices in the same orientation as ``fen_to_board_tensor``.

``MoveEncoder`` indexes squares in absolute chess coordinates (white's north is
+rank). The board tensor mirrors the board when Black is side to move, so a
spatial 8x8x73 head must use that same frame. Flattening ``(8, 8, 73)`` in C
order is ``(row * 8 + col) * 73 + plane``, which is ``chess.square(col, row)``.
"""

from __future__ import annotations

from collections.abc import Iterable

import chess
import numpy as np

from chess_teacher.pipelines.neural_network.board_tensor import oriented_rc
from chess_teacher.pipelines.neural_network.move_encoding import (
    POLICY_VOCAB_SIZE,
    MoveEncoder,
)
from chess_teacher.utils.chess_utils import Color

# Matches the illegal fill used when gathering candidate-slot logits.
ILLEGAL_LOGIT = -1.0e9


def datum_board_flip(color: Color | str) -> bool:
    """True when the board tensor mirrors this side, matching ``pack_board_tensors``."""
    return color != Color.WHITE


def oriented_policy_index(move: chess.Move | str, *, flip: bool) -> int:
    """Map a UCI move to a policy index in the oriented board frame.

    ``flip=False`` is absolute coordinates, so a white move matches
    ``MoveEncoder.encode``. ``flip=True`` mirrors file and rank first, which
    turns Black's pawn push into a +rank step in the tensor.
    """
    if isinstance(move, str):
        move = chess.Move.from_uci(move)
    from_row, from_col = oriented_rc(move.from_square, flip=flip)
    to_row, to_col = oriented_rc(move.to_square, flip=flip)
    oriented = chess.Move(
        chess.square(from_col, from_row),
        chess.square(to_col, to_row),
        promotion=move.promotion,
    )
    return MoveEncoder.encode(oriented)


def oriented_legal_mask(legal_ucis: Iterable[str], *, flip: bool) -> np.ndarray:
    """Float mask ``(4672,)`` with 1 on legal moves in the oriented frame."""
    mask = np.zeros((POLICY_VOCAB_SIZE,), dtype=np.float32)
    for uci in legal_ucis:
        try:
            mask[oriented_policy_index(str(uci), flip=flip)] = 1.0
        except ValueError:
            continue
    return mask
