"""Parent handle for the play worker. No TensorFlow and no real child."""

from __future__ import annotations

import subprocess
import sys

import chess
import pytest

from chess_teacher.bots.move_analysis import BotMoveAnalysis, empty_bot_move_analysis
from chess_teacher.bots.process_bot import ProcessNeuralBot
from chess_teacher.utils.pipeline_utils.step_process import StepProcessError


class _FakeProcess:
    def __init__(self, *, fail_load: bool = False) -> None:
        self.fail_load = fail_load
        self.started = False
        self.closed = False
        self.calls: list[tuple[str, dict[str, object]]] = []

    def start(self) -> None:
        self.started = True

    @property
    def running(self) -> bool:
        return self.started and not self.closed

    def call(self, op: str, **payload: object) -> dict[str, object]:
        self.calls.append((op, payload))
        if op == "load":
            if self.fail_load:
                raise StepProcessError("load failed")
            return {"name": "Personal v3"}
        board = payload["board"]
        assert isinstance(board, chess.Board)
        move = chess.Move.from_uci("e7e5")
        analysis = empty_bot_move_analysis(board, move, temperature_used=0.4)
        return {"uci": move.uci(), "analysis": analysis, "stack": len(board.move_stack)}

    def close(self) -> None:
        self.closed = True


def test_load_keeps_name_temperature_and_does_not_import_tensorflow() -> None:
    process = _FakeProcess()
    bot = ProcessNeuralBot(
        model_uri="models:/baseline/1",
        version="v3",
        display_name="Personal v3",
        temperature=0.4,
        process=process,  # type: ignore[arg-type]
    )
    assert process.started is True
    assert bot.name == "Personal v3"
    assert bot.temperature == pytest.approx(0.4)
    assert bot.running is True
    assert process.calls[0][0] == "load"
    assert process.calls[0][1]["temperature"] == pytest.approx(0.4)
    assert "on_progress" not in process.calls[0][1]
    bot.close()
    bot.close()
    assert process.closed is True
    assert bot.running is False


def test_choose_move_sends_the_board_and_stores_analysis() -> None:
    process = _FakeProcess()
    bot = ProcessNeuralBot(
        model_uri="models:/baseline/1",
        version="v1",
        temperature=0.0,
        process=process,  # type: ignore[arg-type]
    )
    board = chess.Board()
    board.push_san("e4")
    move = bot.choose_move(board)
    assert move.uci() == "e7e5"
    assert process.calls[1][1]["board"] is board
    assert isinstance(bot.last_move_analysis, BotMoveAnalysis)
    assert bot.last_move_analysis.played_uci == "e7e5"
    bot.close()


def test_failed_load_closes_the_child() -> None:
    process = _FakeProcess(fail_load=True)
    with pytest.raises(StepProcessError, match="load failed"):
        ProcessNeuralBot(
            model_uri="models:/baseline/1",
            process=process,  # type: ignore[arg-type]
        )
    assert process.closed is True


def test_process_bot_and_presets_do_not_import_tensorflow() -> None:
    code = (
        "import sys\n"
        "import chess_teacher.bots.process_bot\n"
        "import chess_teacher.bots.presets\n"
        "import chess_teacher.bots.play_worker\n"
        "raise SystemExit(1 if 'tensorflow' in sys.modules else 0)\n"
    )
    completed = subprocess.run([sys.executable, "-c", code], check=False)
    assert completed.returncode == 0, completed.stderr
