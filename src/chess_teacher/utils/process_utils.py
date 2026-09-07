"""Lightweight helpers for multiprocessing-aware code paths."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from chess_teacher.utils.env_utils import get_environment, get_hostname

if TYPE_CHECKING:
    from chess_teacher.utils.logging.logger import EnhancedLogger

try:
    import resource as _resource
except ImportError:  # pragma: no cover - Windows
    _resource = None


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


def _cpu_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def _affinity_count() -> int | None:
    if hasattr(os, "sched_getaffinity"):
        try:
            return len(os.sched_getaffinity(0))
        except OSError:
            return None
    return None


def _load1() -> float | None:
    try:
        return float(os.getloadavg()[0])
    except (AttributeError, OSError):
        return None


def _parse_proc_kb(path: Path, key: str) -> float | None:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith(key):
                parts = line.split()
                if len(parts) >= 2:
                    return float(parts[1])
    except OSError:
        return None
    return None


def _rss_mb() -> float:
    rss_kb = _parse_proc_kb(Path("/proc/self/status"), "VmRSS:")
    if rss_kb is not None:
        return rss_kb / 1024.0
    if _resource is not None:
        rss = float(_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return rss / (1024.0 * 1024.0)
        return rss / 1024.0
    return 0.0


def _meminfo_mb(key: str) -> float | None:
    kb = _parse_proc_kb(Path("/proc/meminfo"), key)
    if kb is None:
        return None
    return kb / 1024.0


@dataclass(frozen=True, slots=True)
class HostPressure:
    """Point-in-time process + host resource snapshot for heavy-step logs."""

    rss_mb: float
    cpu_count: int
    affinity_count: int | None
    load1: float | None
    mem_available_mb: float | None
    mem_total_mb: float | None

    def format_fields(self) -> str:
        parts = [f"rss_mb={self.rss_mb:.1f}", f"cpu_count={self.cpu_count}"]
        if self.affinity_count is not None:
            parts.append(f"affinity={self.affinity_count}")
        if self.load1 is not None:
            parts.append(f"load1={self.load1:.2f}")
        if self.mem_available_mb is not None:
            parts.append(f"mem_available_mb={self.mem_available_mb:.0f}")
        if self.mem_total_mb is not None:
            parts.append(f"mem_total_mb={self.mem_total_mb:.0f}")
        return " ".join(parts)


def snapshot_host_pressure() -> HostPressure:
    """Capture RSS, CPU, load, and available memory without extra dependencies."""
    return HostPressure(
        rss_mb=_rss_mb(),
        cpu_count=_cpu_count(),
        affinity_count=_affinity_count(),
        load1=_load1(),
        mem_available_mb=_meminfo_mb("MemAvailable:"),
        mem_total_mb=_meminfo_mb("MemTotal:"),
    )


def _format_extra_fields(fields: dict[str, object]) -> str:
    return " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)


def log_script_runtime_context(logger: EnhancedLogger, *, script: str) -> None:
    """Log standard runtime context for orchestrated K8s entrypoint scripts."""
    pressure = snapshot_host_pressure()
    logger.info(
        "%s runtime context environment=%s hostname=%s %s",
        script,
        get_environment() or "unknown",
        get_hostname() or "unknown",
        pressure.format_fields(),
    )


@contextmanager
def log_heavy_operation(
    logger: Any,
    operation: str,
    /,
    **fields: object,
) -> Iterator[HostPressure]:
    """Log start/finish (or failure) of a host-heavy operation with duration and RSS."""
    extra = _format_extra_fields(fields)
    label = f"{operation} {extra}".strip() if extra else operation
    started = snapshot_host_pressure()
    t0 = time.monotonic()
    logger.info("%s started %s", label, started.format_fields())
    try:
        yield started
    except Exception:
        ended = snapshot_host_pressure()
        logger.info(
            "%s failed duration_s=%.2f delta_rss_mb=%.1f %s",
            label,
            time.monotonic() - t0,
            ended.rss_mb - started.rss_mb,
            ended.format_fields(),
        )
        raise
    else:
        ended = snapshot_host_pressure()
        logger.info(
            "%s finished duration_s=%.2f delta_rss_mb=%.1f %s",
            label,
            time.monotonic() - t0,
            ended.rss_mb - started.rss_mb,
            ended.format_fields(),
        )


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

    def error(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def exception(self, *args: object, **kwargs: object) -> None:
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
                from chess_teacher.utils.logging.config import get_logger

                self._logger = get_logger(self._name)
        return self._logger

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get(), name)
