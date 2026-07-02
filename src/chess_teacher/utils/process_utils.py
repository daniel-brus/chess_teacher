"""Lightweight helpers for multiprocessing-aware code paths."""

from __future__ import annotations

from typing import NoReturn


def is_parent_process() -> bool:
    """True in the main interpreter process (not a spawned worker child)."""
    from multiprocessing import parent_process

    return parent_process() is None


class _WorkerNoOpLogger:
    """Logger stand-in for worker processes; never touches app logging setup."""

    def info(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def warning(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def log_and_raise(self, exc: Exception, *args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise exc


# Global singleton instance of the worker no-op logger.
WORKER_NO_OP_LOGGER: _WorkerNoOpLogger = _WorkerNoOpLogger()
