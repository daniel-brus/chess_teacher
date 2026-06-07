import inspect
import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, NoReturn, cast

from chess_teacher.utils.env_utils import get_env_variable
from chess_teacher.utils.exception_utils import ConfigError
from chess_teacher.utils.log_shipping import (
    SegmentFileHandler,
    get_log_buffer_dir,
    register_log_shutdown_hooks,
    register_segment_handler,
    reset_log_shipping,
    start_log_shipping,
)

_logging_configured = False


def _get_log_dir() -> Path:
    """Get the log buffer directory path from env or default."""
    return get_log_buffer_dir()


class _JsonLinesFormatter(logging.Formatter):
    """Minimal JSON lines formatter."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "log_id": str(uuid.uuid4()),
            "environment": get_env_variable("ENVIRONMENT"),
        }

        if record.exc_info:
            exc_type, exc_value, _ = record.exc_info
            payload["exc_type"] = exc_type.__name__ if exc_type else None
            payload["exc_msg"] = str(exc_value) if exc_value else None
            payload["traceback"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


class _ConsoleFormatter(logging.Formatter):
    """Simple colored console formatter."""

    COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET: ClassVar[str] = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        timestamp = datetime.fromtimestamp(record.created, UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        return (
            f"{timestamp} {color}{record.levelname:<8}{self.RESET} "
            f"{record.name}: {record.getMessage()}"
        )


class EnhancedLogger(logging.Logger):
    """Custom logger with added Exception functionality."""

    def log_and_raise(
        self,
        exc: Exception,
        message: str | None = None,
        level: str = "error",
        include_traceback: bool = True,
    ) -> NoReturn:
        """
        Log an exception message at the specified level and then re-raise it.

        Args:
            exc: The exception to log and raise.
            message: Optional custom message, to override the exception's in the logs.
            level: The logging level (e.g., "error", "warning").
            include_traceback: Whether to include the traceback in the log.
        """
        log_message = message or str(exc)

        try:
            log_fn = getattr(self, level.lower())
        except Exception as e:
            # If the log level is invalid, log the error and raise a ConfigError
            self.error(f"Invalid log level: {level}. Error: {e}", exc_info=True)
            raise ConfigError(f"Invalid log level: {level}.") from e

        log_fn(log_message, exc_info=include_traceback)
        raise exc


def configure_logging(
    *,
    level: str = "INFO",
    log_dir: Path | None = None,
    force: bool = False,
) -> None:
    """Configure application logging with console and buffered segment file output."""

    global _logging_configured

    if _logging_configured and not force:
        return

    if force:
        reset_log_shipping()

    logging.setLoggerClass(EnhancedLogger)

    resolved_level = level.upper()
    buffer_dir = log_dir or _get_log_dir()

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(resolved_level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(resolved_level)
    console_handler.setFormatter(_ConsoleFormatter())
    root.addHandler(console_handler)

    file_handler = SegmentFileHandler(buffer_dir)
    file_handler.setLevel(resolved_level)
    file_handler.setFormatter(_JsonLinesFormatter())
    root.addHandler(file_handler)

    register_segment_handler(file_handler)
    start_log_shipping(buffer_dir)
    register_log_shutdown_hooks()

    _logging_configured = True


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
