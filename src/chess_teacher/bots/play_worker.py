"""Child interpreter for one neural play game.

Stdout is the pickle protocol. Logging goes to stderr. Exiting this process
is what returns the Keras allocation to the host.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

import chess

_PARENT_POLL_SEC = 1.0


def _parent_process_gone(parent_pid: int) -> bool:
    """True when this process has been reparented.

    A parent pid of 1 is a live parent when Streamlit is PID 1 in the pod.
    """
    return os.getppid() != parent_pid


def _watch_parent_process(
    parent_pid: int,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """SIGKILL this process after the parent process exits.

    ``prctl(PR_SET_PDEATHSIG)`` fires when the creating thread exits, not when
    the parent process exits. Streamlit runs the script on a thread that ends
    after each rerun, so that signal killed the worker between moves.

    SIGTERM is not enough. Importing the package installs a SIGTERM handler
    that only flushes logs. SIGKILL cannot be caught.
    """
    while not _parent_process_gone(parent_pid):
        sleep(_PARENT_POLL_SEC)
    os.kill(os.getpid(), signal.SIGKILL)


def _arm_parent_death() -> None:
    """Start the parent-process watch from this process."""
    parent = os.getppid()
    threading.Thread(
        target=_watch_parent_process,
        args=(parent,),
        name="play-worker-parent",
        daemon=True,
    ).start()


class PlayWorker:
    """Holds the in-process bot for the ``load`` and ``choose_move`` ops."""

    def __init__(self, *, bot_factory: Any | None = None) -> None:
        self._bot: Any | None = None
        self._bot_factory = bot_factory

    def load(
        self,
        *,
        model_uri: str,
        version: str | None = None,
        display_name: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, str]:
        if self._bot_factory is not None:
            bot = self._bot_factory(
                model_uri=model_uri,
                version=version,
                display_name=display_name,
                temperature=temperature,
            )
        else:
            from chess_teacher.bots.neural_baseline_bot import NeuralBaselineBot

            bot = NeuralBaselineBot(
                model_uri=model_uri,
                version=version,
                display_name=display_name,
                temperature=temperature,
            )
        self._bot = bot
        name = getattr(bot, "name", None)
        return {"name": name if isinstance(name, str) and name else "NeuralBaseline"}

    def choose_move(self, *, board: chess.Board) -> dict[str, Any]:
        bot = self._bot
        if bot is None:
            raise RuntimeError("choose_move requires load")
        move = bot.choose_move(board)
        return {"uci": move.uci(), "analysis": bot.last_move_analysis}

    def close(self) -> None:
        bot = self._bot
        self._bot = None
        if bot is not None:
            bot.close()


def main() -> None:
    """Serve load and choose_move until the parent asks the process to exit."""
    _arm_parent_death()
    protocol_out = sys.stdout.buffer
    sys.stdout = sys.stderr
    from chess_teacher.utils.pipeline_utils.step_process import serve

    worker = PlayWorker()
    try:
        serve(
            {"load": worker.load, "choose_move": worker.choose_move},
            protocol_out,
        )
    finally:
        worker.close()


if __name__ == "__main__":
    main()
