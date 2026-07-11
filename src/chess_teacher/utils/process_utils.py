"""Lightweight helpers for multiprocessing-aware code paths."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NoReturn

if TYPE_CHECKING:
    from chess_teacher.utils.logging.logger import EnhancedLogger


def is_parent_process() -> bool:
    """True in the main interpreter process (not a spawned worker child)."""
    from multiprocessing import current_process, parent_process

    if parent_process() is not None:
        return False

    # During Windows spawn, parent_process() is still None while the main script
    # is re-imported, but the process name is already SpawnProcess-* / SpawnPoolWorker-*.
    if current_process().name != "MainProcess":
        return False

    return True


def run_script_main(main: Callable[[], int | None]) -> None:
    """Call from ``if __name__ == "__main__"`` blocks in executable scripts.

    On Windows spawn, worker processes re-import the entry script as ``__main__``.
    This helper no-ops in those workers so only the real parent runs ``main()``.
    """
    if not is_parent_process():
        return
    sys.exit(main() or 0)


class _WorkerNoOpLogger:
    """Logger stand-in for worker processes; never touches app logging setup."""

    def info(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def debug(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def warning(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def log_and_raise(self, exc: Exception, *args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise exc


# Global singleton instance of the worker no-op logger.
WORKER_NO_OP_LOGGER: _WorkerNoOpLogger = _WorkerNoOpLogger()


class WorkerSafeLogger:
    """Lazy logger: real logger in the parent process, no-op in pool workers."""

    def __init__(self, name: str | None = None) -> None:
        self._name = name
        self._logger: EnhancedLogger | _WorkerNoOpLogger | None = None

    def _get(self) -> EnhancedLogger | _WorkerNoOpLogger:
        if self._logger is None:
            if not is_parent_process():
                self._logger = WORKER_NO_OP_LOGGER
            else:
                from chess_teacher.utils.logging import get_logger

                self._logger = get_logger(self._name)
        return self._logger

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get(), name)
