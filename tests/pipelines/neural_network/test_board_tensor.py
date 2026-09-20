"""Unit tests for Phase 2c board tensors."""

from __future__ import annotations

import chess
import numpy as np

from chess_teacher.pipelines.neural_network.board_tensor import (
    BOARD_TENSOR_CHANNELS,
    BOARD_TENSOR_SHAPE,
    BOARD_TENSOR_VERSION,
    board_plane_names,
    fen_to_board_tensor,
)


def test_board_plane_names_match_channel_count() -> None:
    assert len(board_plane_names()) == BOARD_TENSOR_CHANNELS
    assert BOARD_TENSOR_VERSION == 1
    assert BOARD_TENSOR_SHAPE == (8, 8, BOARD_TENSOR_CHANNELS)


def test_startpos_white_our_pawns_on_rank_1() -> None:
    """Side-to-move white: our pawns on tensor row 1 (rank 2)."""
    fen = chess.STARTING_FEN
    t = fen_to_board_tensor(fen, color_is_white=True)
    assert t.shape == BOARD_TENSOR_SHAPE
    assert t.dtype == np.float32
    # Our pawns = channel 0; white pawns on rank 2 → row 1
    assert float(np.sum(t[1, :, 0])) == 8.0
    # Their pawns = channel 6 on rank 7 → row 6
    assert float(np.sum(t[6, :, 6])) == 8.0
    # Castling: both sides have K+Q
    assert float(t[0, 0, 12]) == 1.0
    assert float(t[0, 0, 13]) == 1.0
    assert float(t[0, 0, 14]) == 1.0
    assert float(t[0, 0, 15]) == 1.0


def test_black_to_move_flips_so_our_pieces_bottom() -> None:
    board = chess.Board()
    board.push_san("e4")
    # Black to move - our (Black) pieces should sit on rows 0-1 after flip.
    t = fen_to_board_tensor(board.fen(), color_is_white=False)
    # Black pawns originally rank 7; after flip -> row 1
    assert float(np.sum(t[1, :, 0])) == 8.0
    # White pawns originally rank 2; after flip -> row 6 (with e-pawn advanced: 7 on row 6, 1 elsewhere)
    assert float(np.sum(t[:, :, 6])) == 8.0


def test_en_passant_plane() -> None:
    board = chess.Board()
    board.push_san("e4")
    board.push_san("a6")
    board.push_san("e5")
    board.push_san("d5")  # creates EP on d6
    assert board.ep_square is not None
    t = fen_to_board_tensor(board.fen(), color_is_white=True)
    assert float(np.sum(t[:, :, 16])) == 1.0
