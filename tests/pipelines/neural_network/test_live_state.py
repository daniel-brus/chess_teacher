"""Live state encoding without Stockfish (eval passed in)."""

from __future__ import annotations

import chess
import numpy as np

from chess_teacher.pipelines.neural_network.live_state import (
    _opponent_move_flags,
    compose_live_state_vector,
)


def test_compose_live_state_finite_and_black_to_move_after_e4() -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    live = compose_live_state_vector(
        board,
        evaluation_white_pov=0.35,
        last_opponent_move_uci="e2e4",
    )
    assert live.dtype == np.float32
    assert live.ndim == 1
    assert np.isfinite(live).all()
    # color_is_white flag is index 6 (after 6 opponent piece flags)
    assert live[6] == np.float32(0.0)  # Black to move


def test_opponent_move_flags_capture_from_stack() -> None:
    board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2")
    board.push_uci("e4d5")
    flags = _opponent_move_flags(board, "e4d5")
    assert flags["opponent_move_was_capture"] is True
    assert flags["opponent_move_was_pawn"] is True


def test_opponent_move_flags_quiet_move() -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    flags = _opponent_move_flags(board, "e2e4")
    assert flags["opponent_move_was_capture"] is False
    assert flags["opponent_move_was_pawn"] is True
