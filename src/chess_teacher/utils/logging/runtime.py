"""Process/thread lifecycle for logging setup, shutdown, and shipping."""

from __future__ import annotations

import atexit
import logging
import signal
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from chess_teacher.utils.logging.buffer import SegmentFileHandler
from chess_teacher.utils.process_utils import is_parent_process

if TYPE_CHECKING:
    from chess_teacher.utils.logging.shipping import LogShipper

_configure_lock = threading.Lock()
_logging_configured = False

# Shipper is not imported at module load: shipping → object_storage sits above
# core logging in the utils DAG. Import shipping only inside ship lifecycle fns.
_shipper: LogShipper | None = None
_segment_handler: SegmentFileHandler | None = None
_shutdown_registered = False
_shutdown_done = False
_shutdown_lock = threading.Lock()


def is_configured() -> bool:
    return _logging_configured


def mark_configured() -> None:
    global _logging_configured
    _logging_configured = True


def clear_configured() -> None:
    global _logging_configured
    _logging_configured = False


def configure_lock() -> threading.Lock:
    return _configure_lock


def attach_handlers(
    *,
    level: str,
    buffer_dir: Path,
    console_formatter: logging.Formatter,
    file_formatter: logging.Formatter,
) -> None:
    """Wire console and segment file handlers on the root logger."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    root.addHandler(console_handler)

    if is_parent_process():
        file_handler = SegmentFileHandler(buffer_dir)
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        root.addHandler(file_handler)

        if file_handler.owns_active_log:
            register_segment_handler(file_handler)
            register_shutdown_hooks()
        if file_handler.is_primary_writer:
            start_log_shipping(buffer_dir)


def start_log_shipping(buffer_dir: Path) -> LogShipper | None:
    """Start the background log shipper if enabled."""
    global _shipper
    from chess_teacher.utils.logging.shipping import LogShipper, is_log_ship_enabled

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


def register_shutdown_hooks() -> None:
    """Register process shutdown hooks once."""
    global _shutdown_registered
    if _shutdown_registered:
        return
    _shutdown_registered = True

    def _handle_signal(signum: int, frame: object | None) -> None:
        shutdown_logging()

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
            from chess_teacher.utils.logging.shipping import SHIP_SHUTDOWN_TIMEOUT_SECONDS

            _shipper.stop_and_drain(SHIP_SHUTDOWN_TIMEOUT_SECONDS)
            _shipper = None


def reset_log_shipping() -> None:
    """Reset runtime state (for tests)."""
    global _shipper, _segment_handler, _shutdown_registered, _shutdown_done
    if _shipper is not None:
        _shipper.stop()
    if _segment_handler is not None:
        _segment_handler.close()
    _shipper = None
    _segment_handler = None
    _shutdown_registered = False
    _shutdown_done = False
