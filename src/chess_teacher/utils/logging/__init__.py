"""Application logging: configuration, buffering, and shipping.

Core logging (logger, config, buffer) is a low utils layer.
Shipping talks to object storage and must not be imported by core at module load —
import ``chess_teacher.utils.logging.shipping`` or use the lazy re-exports below.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chess_teacher.utils.logging.buffer import (
    LogBufferWriterLock,
    SegmentFileHandler,
    get_log_buffer_dir,
)
from chess_teacher.utils.logging.config import configure_logging, get_logger
from chess_teacher.utils.logging.logger import EnhancedLogger
from chess_teacher.utils.logging.runtime import (
    register_segment_handler,
    register_shutdown_hooks,
    reset_log_shipping,
    shutdown_logging,
    start_log_shipping,
)

if TYPE_CHECKING:
    from chess_teacher.utils.logging.shipping import LogShipper

__all__ = [
    "EnhancedLogger",
    "LogBufferWriterLock",
    "LogShipper",
    "SegmentFileHandler",
    "configure_logging",
    "get_log_buffer_dir",
    "get_logger",
    "log_storage_key_for_segment",
    "register_segment_handler",
    "register_shutdown_hooks",
    "reset_log_shipping",
    "shutdown_logging",
    "start_log_shipping",
]

_SHIPPING_EXPORTS = frozenset({"LogShipper", "log_storage_key_for_segment"})


def __getattr__(name: str) -> Any:
    if name in _SHIPPING_EXPORTS:
        from chess_teacher.utils.logging import shipping as _shipping

        return getattr(_shipping, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
