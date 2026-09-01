from __future__ import annotations

import chess
import pytest

from chess_teacher.bots import RandomBot, get_bot_preset
from streamlit_utils.play_game import (
    apply_bot_move,
    apply_uci_move,
    game_status_message,
    resolve_user_color,
    start_new_game,
)


def test_resolve_user_color_fixed() -> None:
    assert resolve_user_color("White") == chess.WHITE
    assert resolve_user_color("black") == chess.BLACK


def test_random_bot_chooses_legal_move() -> None:
    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    move = RandomBot(seed=0).choose_move(board)
    assert move in board.legal_moves


def test_start_new_game_black_waits_for_bot() -> None:
    state = start_new_game("Black", "random")
    assert state.user_color == chess.BLACK
    assert state.pending_bot_move is True


def test_apply_legal_move_switches_turn() -> None:
    state = start_new_game("White", "random")
    next_state = apply_uci_move(state, "e2e4")
    assert next_state.board.turn == chess.BLACK
    assert next_state.last_move_uci == "e2e4"
    assert next_state.pending_bot_move is True


def test_apply_bot_move_after_user_move() -> None:
    state = start_new_game("White", "random")
    state = apply_uci_move(state, "e2e4")
    bot = RandomBot(seed=1)
    bot_state = apply_bot_move(state, bot)
    assert bot_state.board.fullmove_number == 2
    assert bot_state.board.turn == chess.WHITE
    assert bot_state.pending_bot_move is False
    assert bot_state.last_move_uci is not None


def test_game_status_message_resignation() -> None:
    state = start_new_game("White", "random")
    state.resigned = True
    assert game_status_message(state) == "You resigned. ChessBot wins."


def test_get_bot_preset_unknown() -> None:
    with pytest.raises(KeyError, match="Unknown bot preset"):
        get_bot_preset("missing")
