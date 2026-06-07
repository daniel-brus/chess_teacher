"""Upload closed local log segments to object storage."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from chess_teacher.utils.env_utils import get_env_variable
from chess_teacher.utils.logging.buffer import LOG_STORAGE_PREFIX, READY_SUFFIX

SHIP_INTERVAL_SECONDS = 60.0
SHIP_SHUTDOWN_TIMEOUT_SECONDS = 20.0

_logger = logging.getLogger(__name__)


def log_storage_key_for_segment(segment_ready_path: Path, buffer_dir: Path) -> str:
    """Map a local ``*.ready`` segment path to an object storage key under ``STORAGE_ROOT``."""
    relative = segment_ready_path.relative_to(buffer_dir)
    log_relative = Path(str(relative).removesuffix(READY_SUFFIX))
    return f"{LOG_STORAGE_PREFIX}/{log_relative.as_posix()}"


def is_log_ship_enabled() -> bool:
    """Return whether the in-process log shipper should run."""
    explicit = get_env_variable("LOG_SHIP_ENABLED", default="")
    if explicit == "":
        return get_env_variable("STORAGE_BACKEND") == "s3"
    return explicit.lower() in {"1", "true", "yes"}


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
        for ready_path in closed_dir.rglob(f"*{READY_SUFFIX}"):
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
