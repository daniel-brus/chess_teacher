"""Parent-side handle for a neural bot running in a child interpreter.

The Streamlit process must not import TensorFlow. The child module name is a
string so importing this file does not load ``play_worker``.
"""

from __future__ import annotations

from typing import Any, Protocol

import chess

from chess_teacher.bots.base import ChessBot
from chess_teacher.bots.move_analysis import BotMoveAnalysis
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.pipeline_utils.step_process import StepProcess, StepProcessError

logger = get_logger()

PLAY_WORKER_MODULE = "chess_teacher.bots.play_worker"


class PlayWorkerProcess(Protocol):
    """The slice of ``StepProcess`` this handle uses. Tests pass a fake."""

    @property
    def running(self) -> bool: ...

    def start(self) -> None: ...

    def call(self, op: str, **payload: Any) -> Any: ...

    def close(self) -> None: ...


class ProcessNeuralBot(ChessBot):
    """One child process for one neural game. ``close`` exits that process."""

    def __init__(
        self,
        *,
        model_uri: str,
        version: str | None = None,
        display_name: str | None = None,
        temperature: float = 0.0,
        process: PlayWorkerProcess | None = None,
    ) -> None:
        self.model_uri = model_uri
        self.version = version
        self.temperature = float(temperature)
        self.name = display_name or (f"Baseline {version}" if version else "NeuralBaseline")
        self.last_move_analysis: BotMoveAnalysis | None = None
        self._closed = False
        self._process: PlayWorkerProcess | None = process or StepProcess(PLAY_WORKER_MODULE)
        self._process.start()
        try:
            loaded = self._process.call(
                "load",
                model_uri=model_uri,
                version=version,
                display_name=display_name,
                temperature=self.temperature,
            )
        except Exception:
            self.close()
            raise
        if isinstance(loaded, dict):
            name = loaded.get("name")
            if isinstance(name, str) and name:
                self.name = name
        logger.info(
            "Play worker loaded name=%s version=%s",
            self.name,
            version,
        )

    @property
    def running(self) -> bool:
        process = self._process
        return process is not None and not self._closed and process.running

    def choose_move(self, board: chess.Board) -> chess.Move:
        process = self._process
        if process is None or self._closed:
            raise StepProcessError("play worker is not running")
        result: Any = process.call("choose_move", board=board)
        if not isinstance(result, dict):
            raise StepProcessError("choose_move returned an invalid response")
        uci = result.get("uci")
        if not isinstance(uci, str) or not uci:
            raise StepProcessError("choose_move did not return a UCI")
        analysis = result.get("analysis")
        self.last_move_analysis = analysis if isinstance(analysis, BotMoveAnalysis) else None
        return chess.Move.from_uci(uci)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        self._process = None
        if process is not None:
            process.close()
        logger.info("Play worker closed name=%s version=%s", self.name, self.version)
