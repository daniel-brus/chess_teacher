"""Board spatial tensors for Phase 2c hybrid encoder (offline).

Builds fixed-shape ``(8, 8, C)`` float32 planes from ``fen_before``, oriented so
side-to-move ("us") sits on the bottom ranks. See
``.agents/docs/ml-phase2c-board-encoder.md`` for plane layout + version rules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import chess
import numpy as np

if TYPE_CHECKING:
    from chess_teacher.pipelines.neural_network.create_training_set import TrainingDatum

# Bump when channel order, count, or orientation policy changes.
BOARD_TENSOR_VERSION = 1

# Plane indices (must match research note).
_PLANE_OUR_PIECES = (0, 1, 2, 3, 4, 5)  # P N B R Q K
_PLANE_THEIR_PIECES = (6, 7, 8, 9, 10, 11)
_PLANE_OUR_CASTLE_K = 12
_PLANE_OUR_CASTLE_Q = 13
_PLANE_THEIR_CASTLE_K = 14
_PLANE_THEIR_CASTLE_Q = 15
_PLANE_EN_PASSANT = 16

BOARD_TENSOR_CHANNELS = 17
BOARD_TENSOR_SHAPE = (8, 8, BOARD_TENSOR_CHANNELS)

_PIECE_ORDER: tuple[chess.PieceType, ...] = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)


def _square_to_rc(square: int, *, flip: bool) -> tuple[int, int]:
    """Map chess square → (row, col) with row 0 = our back rank after orientation."""
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    if flip:
        rank = 7 - rank
        file = 7 - file
    # Tensor row 0 = rank 0 from our POV (our back rank).
    return rank, file


def fen_to_board_tensor(
    fen: str,
    *,
    color_is_white: bool | None = None,
) -> np.ndarray:
    """Encode ``fen`` as ``(8, 8, C)`` float32.

    If ``color_is_white`` is None, use side-to-move from the FEN.
    When Black is "us", the board is mirrored so us pieces sit on ranks 0-1.
    """
    board = chess.Board(fen)
    us_white = board.turn == chess.WHITE if color_is_white is None else bool(color_is_white)
    flip = not us_white

    out = np.zeros(BOARD_TENSOR_SHAPE, dtype=np.float32)

    for piece_type, our_ch, their_ch in zip(
        _PIECE_ORDER, _PLANE_OUR_PIECES, _PLANE_THEIR_PIECES, strict=True
    ):
        for square in board.pieces(piece_type, chess.WHITE):
            r, c = _square_to_rc(square, flip=flip)
            ch = our_ch if us_white else their_ch
            out[r, c, ch] = 1.0
        for square in board.pieces(piece_type, chess.BLACK):
            r, c = _square_to_rc(square, flip=flip)
            ch = their_ch if us_white else our_ch
            out[r, c, ch] = 1.0

    # Castling rights: fill entire plane with 0/1 (AlphaZero-style constant planes).
    if us_white:
        our_k = board.has_kingside_castling_rights(chess.WHITE)
        our_q = board.has_queenside_castling_rights(chess.WHITE)
        their_k = board.has_kingside_castling_rights(chess.BLACK)
        their_q = board.has_queenside_castling_rights(chess.BLACK)
    else:
        our_k = board.has_kingside_castling_rights(chess.BLACK)
        our_q = board.has_queenside_castling_rights(chess.BLACK)
        their_k = board.has_kingside_castling_rights(chess.WHITE)
        their_q = board.has_queenside_castling_rights(chess.WHITE)

    if our_k:
        out[:, :, _PLANE_OUR_CASTLE_K] = 1.0
    if our_q:
        out[:, :, _PLANE_OUR_CASTLE_Q] = 1.0
    if their_k:
        out[:, :, _PLANE_THEIR_CASTLE_K] = 1.0
    if their_q:
        out[:, :, _PLANE_THEIR_CASTLE_Q] = 1.0

    if board.ep_square is not None:
        r, c = _square_to_rc(board.ep_square, flip=flip)
        out[r, c, _PLANE_EN_PASSANT] = 1.0

    return out


def pack_board_tensors(datums: list[TrainingDatum]) -> np.ndarray:
    """Stack board tensors for a datum list → ``(N, 8, 8, C)``."""
    if not datums:
        return np.zeros((0, *BOARD_TENSOR_SHAPE), dtype=np.float32)
    from chess_teacher.utils.chess_utils import Color

    planes = [
        fen_to_board_tensor(
            d.fen_before,
            color_is_white=d.color == Color.WHITE,
        )
        for d in datums
    ]
    return np.stack(planes, axis=0).astype(np.float32, copy=False)


def board_plane_names() -> tuple[str, ...]:
    """Human-readable channel names for notebooks / debugging."""
    pieces = ("pawn", "knight", "bishop", "rook", "queen", "king")
    names = [f"our_{p}" for p in pieces] + [f"their_{p}" for p in pieces]
    names += [
        "our_castle_k",
        "our_castle_q",
        "their_castle_k",
        "their_castle_q",
        "en_passant",
    ]
    return tuple(names)
