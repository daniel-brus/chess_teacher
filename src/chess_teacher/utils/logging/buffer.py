"""Local log buffer paths, writer ownership, and segment rotation."""

from __future__ import annotations

import atexit
import logging
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

from chess_teacher.utils.env_utils import get_env_variable, get_hostname
from chess_teacher.utils.exception_utils import ConfigError

LOG_STORAGE_PREFIX = "logs/python/buffer"
READY_SUFFIX = ".ready"
_ACTIVE_WRITER_LOCK = ".writer.lock"

SEGMENT_INTERVAL_SECONDS = 600
SEGMENT_MAX_BYTES = 5 * 1024 * 1024


def _require_hostname() -> str:
    hostname = get_hostname()
    if hostname is None:
        raise ConfigError("Missing required environment variable: HOSTNAME")
    return hostname


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class LogBufferWriterLock:
    """Exclusive claim on one active log file within a buffer directory."""

    def __init__(
        self,
        buffer_dir: Path,
        *,
        lock_name: str = _ACTIVE_WRITER_LOCK,
        pid: int | None = None,
    ) -> None:
        self.buffer_dir = Path(buffer_dir)
        self.lock_path = self.buffer_dir / "active" / lock_name
        self.pid = pid if pid is not None else os.getpid()
        self.acquired = False

    def holder_pid(self) -> int | None:
        try:
            return int(self.lock_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None

    def acquire(self) -> bool:
        """Return True when this instance may write to the active log file."""
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        if self._try_create_lock():
            self._mark_acquired()
            return True

        holder = self.holder_pid()
        if holder == self.pid:
            return False

        if holder is not None and _pid_is_alive(holder):
            return False

        self.lock_path.unlink(missing_ok=True)
        if self._try_create_lock():
            self._mark_acquired()
            return True
        return False

    def _try_create_lock(self) -> bool:
        try:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(str(self.pid))
        return True

    def _mark_acquired(self) -> None:
        self.acquired = True
        atexit.register(self.release)

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            if self.holder_pid() == self.pid:
                self.lock_path.unlink(missing_ok=True)
        except OSError:
            pass
        self.acquired = False


def get_log_buffer_dir() -> Path:
    """Return the local buffer root for active and closed log segments."""
    try:
        return Path(get_env_variable("LOG_BUFFER_DIR"))
    except ValueError as e:
        raise ConfigError(str(e)) from e


class SegmentFileHandler(logging.Handler):
    """Write JSON-lines logs to a local active file and rotate closed segments for upload."""

    def __init__(
        self,
        buffer_dir: Path,
        *,
        instance_id: str | None = None,
        max_bytes: int | None = None,
        interval_seconds: int | None = None,
        encoding: str = "utf-8",
        writer_mode: str = "auto",
    ) -> None:
        super().__init__()
        self.buffer_dir = Path(buffer_dir)
        self.instance_id = instance_id or _require_hostname()
        self.max_bytes = max_bytes if max_bytes is not None else SEGMENT_MAX_BYTES
        self.interval_seconds = (
            interval_seconds if interval_seconds is not None else SEGMENT_INTERVAL_SECONDS
        )
        self.encoding = encoding
        self._lock = threading.Lock()
        self._stream: TextIO | None = None
        self._opened_at = time.monotonic()
        self.is_primary_writer = False
        self.owns_active_log = False
        self._segment_name_prefix = "app"
        self._writer_lock: LogBufferWriterLock

        if writer_mode == "auxiliary":
            self._init_auxiliary_writer()
        elif writer_mode == "primary":
            self._init_primary_writer(strict=True)
        elif writer_mode == "auto":
            if not self._init_primary_writer(strict=False):
                holder = self._writer_lock.holder_pid()
                if holder is not None and holder != os.getpid() and _pid_is_alive(holder):
                    self._init_auxiliary_writer()
        else:
            raise ConfigError(f"Unsupported log buffer writer_mode: {writer_mode}")

    def _init_primary_writer(self, *, strict: bool) -> bool:
        """Return True when this handler owns the primary active log file."""
        self.active_path = self.buffer_dir / "active" / "app.log"
        self._writer_lock = LogBufferWriterLock(self.buffer_dir)
        self.owns_active_log = self._writer_lock.acquire()
        self.is_primary_writer = self.owns_active_log
        self._segment_name_prefix = "app"

        if self.owns_active_log:
            self.active_path.parent.mkdir(parents=True, exist_ok=True)
            self._open_active_stream()
            return True

        holder = self._writer_lock.holder_pid()
        if holder is not None and holder != os.getpid():
            if strict:
                raise ConfigError(
                    f"Shared log buffer {self.active_path} is already owned by PID {holder}; "
                    f"this process (PID {os.getpid()}) cannot start. "
                    f"Stop the other process using LOG_BUFFER_DIR={self.buffer_dir}."
                )
            return False
        return False

    def _init_auxiliary_writer(self) -> None:
        """Use a per-process active log when the primary buffer is already owned."""
        pid = os.getpid()
        self.active_path = self.buffer_dir / "active" / f"worker-{pid}.log"
        self._writer_lock = LogBufferWriterLock(
            self.buffer_dir,
            lock_name=f".writer.{pid}.lock",
        )
        self.owns_active_log = self._writer_lock.acquire()
        self.is_primary_writer = False
        self._segment_name_prefix = f"worker-{pid}"
        if not self.owns_active_log:
            raise ConfigError(
                f"Could not acquire auxiliary log buffer for PID {pid} "
                f"under LOG_BUFFER_DIR={self.buffer_dir}."
            )
        self.active_path.parent.mkdir(parents=True, exist_ok=True)
        self._open_active_stream()

    def _open_active_stream(self) -> None:
        self._stream = self.active_path.open("a", encoding=self.encoding)
        self._opened_at = time.monotonic()

    def emit(self, record: logging.LogRecord) -> None:
        if not self.owns_active_log:
            return
        try:
            msg = self.format(record)
            with self._lock:
                stream = self._stream
                if stream is None:
                    return
                stream.write(msg + "\n")
                stream.flush()
                if self._should_rotate():
                    self._rotate_locked()
        except Exception:
            self.handleError(record)

    def _should_rotate(self) -> bool:
        if self.active_path.exists() and self.active_path.stat().st_size == 0:
            return False
        if self.active_path.stat().st_size >= self.max_bytes:
            return True
        return (time.monotonic() - self._opened_at) >= self.interval_seconds

    def _closed_segment_path(self) -> Path:
        now = datetime.now(UTC)
        date_path = now.strftime("%Y/%m/%d")
        timestamp = now.strftime("%H%M%SZ")
        return (
            self.buffer_dir
            / "closed"
            / date_path
            / self.instance_id
            / f"{self._segment_name_prefix}-{timestamp}.log{READY_SUFFIX}"
        )

    def _rotate_locked(self) -> None:
        stream = self._stream
        if stream is not None:
            stream.close()
            self._stream = None

        if not self.active_path.exists() or self.active_path.stat().st_size == 0:
            self._open_active_stream()
            return

        closed_path = self._closed_segment_path()
        closed_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.active_path.replace(closed_path)
        except PermissionError:
            # Another process may still hold app.log open (common on Windows spawn).
            self._open_active_stream()
            return
        self._open_active_stream()

    def close_active_segment(self) -> None:
        """Close and rotate the active segment if it contains data."""
        if not self.owns_active_log:
            return
        with self._lock:
            self._rotate_locked()

    def close(self) -> None:
        with self._lock:
            stream = self._stream
            if stream is not None:
                stream.close()
                self._stream = None
        if self.owns_active_log:
            self._writer_lock.release()
        super().close()
