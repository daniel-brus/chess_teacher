"""Public logging configuration entry points."""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import cast

from chess_teacher.utils.logging.buffer import get_log_buffer_dir
from chess_teacher.utils.logging.formatters import ConsoleFormatter, JsonLinesFormatter
from chess_teacher.utils.logging.logger import EnhancedLogger
from chess_teacher.utils.logging.runtime import (
    attach_handlers,
    clear_configured,
    configure_lock,
    is_configured,
    mark_configured,
    reset_log_shipping,
)


def configure_logging(
    *,
    level: str = "INFO",
    log_dir: Path | None = None,
    force: bool = False,
) -> None:
    """Configure application logging with console and buffered segment file output."""
    with configure_lock():
        if is_configured() and not force:
            return

        if force:
            reset_log_shipping()
            clear_configured()

        logging.setLoggerClass(EnhancedLogger)

        resolved_level = level.upper()
        buffer_dir = log_dir or get_log_buffer_dir()

        attach_handlers(
            level=resolved_level,
            buffer_dir=buffer_dir,
            console_formatter=ConsoleFormatter(),
            file_formatter=JsonLinesFormatter(),
        )
        mark_configured()


def get_logger(name: str | None = None) -> EnhancedLogger:
    """Return a module logger and ensure logging is configured once."""
    configure_logging()

    if name:
        return cast(EnhancedLogger, logging.getLogger(name))

    frame = inspect.currentframe()
    if frame is None or frame.f_back is None:
        return cast(EnhancedLogger, logging.getLogger(__name__))

    caller_globals = frame.f_back.f_globals
    caller_name = caller_globals.get("__name__", __name__)
    return cast(EnhancedLogger, logging.getLogger(caller_name))
