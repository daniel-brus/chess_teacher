"""Play worker ops. The real model stays behind NeuralBaselineBot."""

from __future__ import annotations

import signal

import chess
import pytest

from chess_teacher.bots.move_analysis import empty_bot_move_analysis
from chess_teacher.bots.play_worker import PlayWorker, _arm_parent_death


class _FakeNeural:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.name = kwargs.get("display_name") or "Baseline"
        self.closed = False
        self.last_move_analysis = None
        self.seen_stack = 0

    def choose_move(self, board: chess.Board) -> chess.Move:
        self.seen_stack = len(board.move_stack)
        move = chess.Move.from_uci("e7e5")
        self.last_move_analysis = empty_bot_move_analysis(
            board,
            move,
            temperature_used=float(self.kwargs["temperature"]),  # type: ignore[arg-type]
        )
        return move

    def close(self) -> None:
        self.closed = True


def test_load_and_choose_move_keep_the_move_stack() -> None:
    worker = PlayWorker(bot_factory=_FakeNeural)
    loaded = worker.load(
        model_uri="models:/baseline/1",
        version="v2",
        display_name="Baseline v2",
        temperature=0.2,
    )
    assert loaded == {"name": "Baseline v2"}
    board = chess.Board()
    board.push_san("e4")
    result = worker.choose_move(board=board)
    assert result["uci"] == "e7e5"
    assert result["analysis"].played_uci == "e7e5"
    assert worker._bot is not None
    assert worker._bot.seen_stack == 1
    worker.close()
    assert worker._bot is None


def test_choose_move_before_load_raises() -> None:
    worker = PlayWorker(bot_factory=_FakeNeural)
    with pytest.raises(RuntimeError, match="requires load"):
        worker.choose_move(board=chess.Board())


def _install_prctl(monkeypatch: pytest.MonkeyPatch, prctl) -> None:
    monkeypatch.setattr(
        "chess_teacher.bots.play_worker.ctypes.CDLL",
        lambda *_args, **_kwargs: type("_Lib", (), {"prctl": staticmethod(prctl)})(),
    )


def test_parent_death_uses_sigkill_and_pid_1_stays_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int]] = []
    killed: list[int] = []

    def prctl(option: int, sig: int) -> int:
        calls.append((option, sig))
        return 0

    monkeypatch.setattr("chess_teacher.bots.play_worker.os.getppid", lambda: 1)
    monkeypatch.setattr(
        "chess_teacher.bots.play_worker.os.kill",
        lambda _pid, sig: killed.append(sig),
    )
    _install_prctl(monkeypatch, prctl)
    _arm_parent_death()
    assert calls == [(1, int(signal.SIGKILL))]
    assert killed == []


def test_parent_death_kills_if_reparented_during_prctl(monkeypatch: pytest.MonkeyPatch) -> None:
    ppid = {"value": 40}
    killed: list[int] = []

    def prctl(_option: int, _sig: int) -> int:
        ppid["value"] = 1
        return 0

    monkeypatch.setattr("chess_teacher.bots.play_worker.os.getppid", lambda: ppid["value"])
    monkeypatch.setattr(
        "chess_teacher.bots.play_worker.os.kill",
        lambda _pid, sig: killed.append(sig),
    )
    _install_prctl(monkeypatch, prctl)
    _arm_parent_death()
    assert killed == [int(signal.SIGKILL)]
