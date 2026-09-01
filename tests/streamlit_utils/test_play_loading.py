from __future__ import annotations

from chess_teacher.bots import stockfish_preset_key
from streamlit_utils.play_loading import (
    MSG_BASELINE_PREPARE,
    MSG_BASELINE_THINKING,
    MSG_OPPONENT_PREPARE,
    MSG_STOCKFISH_ENGINE,
    bot_thinking_message,
    initial_bot_loading_message,
)


def test_initial_bot_loading_message_baseline() -> None:
    assert initial_bot_loading_message("baseline:v1") == MSG_BASELINE_PREPARE


def test_initial_bot_loading_message_stockfish() -> None:
    assert initial_bot_loading_message(stockfish_preset_key(5)) == MSG_STOCKFISH_ENGINE


def test_initial_bot_loading_message_other() -> None:
    assert initial_bot_loading_message("random") == MSG_OPPONENT_PREPARE


def test_bot_thinking_message_baseline() -> None:
    msg = bot_thinking_message("baseline:v1", label="Baseline v1")
    assert "Baseline v1" in msg
    assert MSG_BASELINE_THINKING in msg


def test_bot_thinking_message_stockfish() -> None:
    assert bot_thinking_message(stockfish_preset_key(3), label="Stockfish") == (
        "Stockfish is thinking…"
    )
