"""Buffered local log segments with optional upload to object storage."""

from __future__ import annotations

import atexit
import logging
import os
import signal
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from chess_teacher.utils.env_utils import get_env_variable, get_hostname
from chess_teacher.utils.exception_utils import ConfigError

LOG_STORAGE_PREFIX = "logs/python/buffer"
_READY_SUFFIX = ".ready"
_ACTIVE_WRITER_LOCK = ".writer.lock"

SEGMENT_INTERVAL_SECONDS = 600
SEGMENT_MAX_BYTES = 5 * 1024 * 1024
SHIP_INTERVAL_SECONDS = 60.0
SHIP_SHUTDOWN_TIMEOUT_SECONDS = 20.0

_shipper: LogShipper | None = None
_segment_handler: SegmentFileHandler | None = None
_shutdown_registered = False
_shutdown_done = False
_shutdown_lock = threading.Lock()

_logger = logging.getLogger(__name__)


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
    """Exclusive claim on the shared active log file for one buffer directory."""

    def __init__(self, buffer_dir: Path, *, pid: int | None = None) -> None:
        self.buffer_dir = Path(buffer_dir)
        self.lock_path = self.buffer_dir / "active" / _ACTIVE_WRITER_LOCK
        self.pid = pid if pid is not None else os.getpid()
        self.acquired = False

    def holder_pid(self) -> int | None:
        try:
            return int(self.lock_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None

    def acquire(self) -> bool:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        if self._try_create_lock():
            self._mark_acquired()
            return True

        holder = self.holder_pid()
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


def log_storage_key_for_segment(segment_ready_path: Path, buffer_dir: Path) -> str:
    """Map a local ``*.ready`` segment path to an object storage key under ``STORAGE_ROOT``."""
    relative = segment_ready_path.relative_to(buffer_dir)
    log_relative = Path(str(relative).removesuffix(_READY_SUFFIX))
    return f"{LOG_STORAGE_PREFIX}/{log_relative.as_posix()}"


def is_log_ship_enabled() -> bool:
    """Return whether the in-process log shipper should run."""
    explicit = get_env_variable("LOG_SHIP_ENABLED", default="")
    if explicit == "":
        return get_env_variable("STORAGE_BACKEND") == "s3"
    return explicit.lower() in {"1", "true", "yes"}


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
        self._stream: object | None = None
        self._opened_at = time.monotonic()
        self.active_path = self.buffer_dir / "active" / "app.log"
        self._writer_lock = LogBufferWriterLock(self.buffer_dir)
        self.owns_active_log = self._writer_lock.acquire()
        if not self.owns_active_log:
            holder = self._writer_lock.holder_pid()
            raise ConfigError(
                f"Shared log buffer {self.active_path} is already owned by PID {holder}; "
                f"this process (PID {os.getpid()}) cannot start. "
                f"Stop the other process using LOG_BUFFER_DIR={self.buffer_dir}."
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
            / f"app-{timestamp}.log{_READY_SUFFIX}"
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
        self.active_path.replace(closed_path)
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


class LogShipper:
    """Background uploader for closed ``*.ready`` log segments."""

    def __init__(
        self,
        buffer_dir: Path,
        *,
        scan_interval_seconds: float | None = None,
    ) -> None:
        self.buffer_dir = Path(buffer_dir)
        self.scan_interval_seconds = (
            scan_interval_seconds if scan_interval_seconds is not None else SHIP_INTERVAL_SECONDS
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="log-shipper", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def stop_and_drain(self, timeout_seconds: float) -> None:
        self._stop.set()
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            pending = self._pending_ready_paths()
            if not pending:
                break
            self.scan_once()
            if not self._pending_ready_paths():
                break
            time.sleep(0.1)
        if self._thread is not None:
            self._thread.join(timeout=max(0.0, deadline - time.monotonic()))
            self._thread = None

    def scan_once(self) -> int:
        uploaded = 0
        for ready_path in self._pending_ready_paths():
            if self._upload_segment(ready_path):
                uploaded += 1
        return uploaded

    def _pending_ready_paths(self) -> list[Path]:
        closed_dir = self.buffer_dir / "closed"
        if not closed_dir.exists():
            return []
        pending: list[Path] = []
        for ready_path in closed_dir.rglob(f"*{_READY_SUFFIX}"):
            pending.append(ready_path)
        pending.sort()
        return pending

    def _upload_segment(self, ready_path: Path) -> bool:
        try:
            data = ready_path.read_bytes()
            key = log_storage_key_for_segment(ready_path, self.buffer_dir)
            from chess_teacher.utils.object_storage.factory import get_raw_storage

            get_raw_storage().write_bytes(key, data, overwrite=False)
            ready_path.unlink()
            return True
        except Exception:
            _logger.warning("Failed to upload log segment %s", ready_path, exc_info=True)
            return False

    def _run(self) -> None:
        while not self._stop.wait(self.scan_interval_seconds):
            self.scan_once()


def start_log_shipping(buffer_dir: Path) -> LogShipper | None:
    """Start the background log shipper if enabled."""
    global _shipper
    if not is_log_ship_enabled():
        return None
    if _shipper is None:
        _shipper = LogShipper(buffer_dir)
        _shipper.start()
    return _shipper


def register_segment_handler(handler: SegmentFileHandler) -> None:
    """Track the active segment handler for shutdown rotation."""
    global _segment_handler
    _segment_handler = handler


def register_log_shutdown_hooks() -> None:
    """Register process shutdown hooks once."""
    global _shutdown_registered
    if _shutdown_registered:
        return
    _shutdown_registered = True

    def _handle_signal(signum: int, frame: object | None) -> None:
        shutdown_logging()

    # Streamlit and other hosts run app code off the main thread; signals are main-thread only.
    if threading.current_thread() is threading.main_thread():
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)
    atexit.register(shutdown_logging)


def shutdown_logging() -> None:
    """Close the active segment and drain pending uploads."""
    global _shutdown_done, _segment_handler, _shipper
    with _shutdown_lock:
        if _shutdown_done:
            return
        _shutdown_done = True

        if _segment_handler is not None:
            _segment_handler.close_active_segment()

        if _shipper is not None:
            _shipper.stop_and_drain(SHIP_SHUTDOWN_TIMEOUT_SECONDS)
            _shipper = None


def reset_log_shipping() -> None:
    """Reset module state (for tests)."""
    global _shipper, _segment_handler, _shutdown_registered, _shutdown_done
    if _shipper is not None:
        _shipper.stop()
    _shipper = None
    _segment_handler = None
    _shutdown_registered = False
    _shutdown_done = False
