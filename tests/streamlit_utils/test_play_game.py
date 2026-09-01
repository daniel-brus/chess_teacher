from __future__ import annotations

import chess
import pytest

from chess_teacher.bots import RandomBot
from chess_teacher.bots.base import ChessBot
from chess_teacher.bots.presets import BotPreset
from streamlit_utils.play_game import (
    apply_bot_move,
    apply_uci_move,
    create_bot,
    resign_game,
    start_new_game,
)


class _RecordingBot(ChessBot):
    name = "Recording"

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def choose_move(self, board: chess.Board) -> chess.Move:
        return next(iter(board.legal_moves))

    def close(self) -> None:
        return None


def test_start_new_game_stores_baseline_temperature() -> None:
    state = start_new_game("White", "baseline:v1", baseline_temperature=1.25)
    assert state.baseline_temperature == pytest.approx(1.25)


def test_baseline_temperature_survives_user_and_bot_moves() -> None:
    state = start_new_game("White", "random", baseline_temperature=0.5)
    state = apply_uci_move(state, "e2e4")
    assert state.baseline_temperature == pytest.approx(0.5)

    bot = RandomBot(seed=0)
    state = apply_bot_move(state, bot)
    assert state.baseline_temperature == pytest.approx(0.5)


def test_baseline_temperature_survives_resignation() -> None:
    state = start_new_game("White", "random", baseline_temperature=0.75)
    resigned = resign_game(state)
    assert resigned.baseline_temperature == pytest.approx(0.75)
    assert resigned.resigned is True


def test_create_bot_emits_progress_for_random_preset() -> None:
    messages: list[str] = []
    bot = create_bot("random", on_progress=messages.append)
    try:
        assert messages[0] == "Loading opponent settings…"
        assert messages[-1] == "Starting Stockfish engine…"
    finally:
        bot.close()


def test_create_bot_passes_temperature_and_progress_to_baseline_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def factory(*, temperature: float = 0.0, on_progress=None) -> ChessBot:
        captured["temperature"] = temperature
        captured["on_progress"] = on_progress
        return _RecordingBot(temperature=temperature)

    preset = BotPreset(
        key="baseline:v_test",
        label="Baseline v_test",
        description="test",
        factory=factory,
    )
    monkeypatch.setattr(
        "streamlit_utils.play_game.get_bot_preset",
        lambda _key, db_client=None: preset,
    )

    progress: list[str] = []

    def record_progress(message: str) -> None:
        progress.append(message)

    bot = create_bot(
        "baseline:v_test",
        baseline_temperature=1.5,
        on_progress=record_progress,
    )
    try:
        assert captured["temperature"] == pytest.approx(1.5)
        assert captured["on_progress"] is record_progress
        assert progress[0] == "Loading opponent settings…"
    finally:
        bot.close()
