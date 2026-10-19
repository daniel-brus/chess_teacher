"""Play worker ops. The real model stays behind NeuralBaselineBot."""

from __future__ import annotations

import signal

import chess
import pytest

from chess_teacher.bots.move_analysis import empty_bot_move_analysis
from chess_teacher.bots.play_worker import PlayWorker, _watch_parent_process


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


def test_parent_watch_stays_alive_when_pid_1_is_the_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    killed: list[int] = []
    sleeps = {"n": 0}

    def sleep(_seconds: float) -> None:
        sleeps["n"] += 1
        if sleeps["n"] >= 2:
            raise StopIteration

    monkeypatch.setattr("chess_teacher.bots.play_worker.os.getppid", lambda: 1)
    monkeypatch.setattr(
        "chess_teacher.bots.play_worker.os.kill",
        lambda _pid, sig: killed.append(sig),
    )
    with pytest.raises(StopIteration):
        _watch_parent_process(1, sleep=sleep)
    assert killed == []


def test_parent_watch_sigkills_when_reparented(monkeypatch: pytest.MonkeyPatch) -> None:
    ppid = {"value": 40}
    killed: list[int] = []

    def sleep(_seconds: float) -> None:
        ppid["value"] = 1

    monkeypatch.setattr("chess_teacher.bots.play_worker.os.getppid", lambda: ppid["value"])
    monkeypatch.setattr(
        "chess_teacher.bots.play_worker.os.kill",
        lambda _pid, sig: killed.append(sig),
    )
    _watch_parent_process(40, sleep=sleep)
    assert killed == [int(signal.SIGKILL)]
