import inspect
import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from chess_teacher.utils.env_utils import get_env_variable
from chess_teacher.utils.exception_utils import ConfigError

# Module-level flag to track logging configuration
_logging_configured = False


def _get_log_dir() -> Path:
    """Get the log directory path from env or default."""
    base = get_env_variable("RAW_DIR")
    if not base:
        raise ConfigError("Missing env var to configure log_dir: RAW_DIR")
    return Path(base + "/logs/python")


def _build_daily_log_path(base_dir: Path) -> Path:
    """
    Example:
        logs/2026/05/08/app.log
    """

    daily_dir = base_dir / datetime.now(UTC).strftime("%Y/%m/%d")

    daily_dir.mkdir(parents=True, exist_ok=True)

    return daily_dir / "app.log"


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

        # Exception info
        if record.exc_info:
            exc_type, exc_value, _ = record.exc_info

            payload["exc_type"] = exc_type.__name__ if exc_type else None

            payload["exc_msg"] = str(exc_value) if exc_value else None

            payload["traceback"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


class _ConsoleFormatter(logging.Formatter):
    """Simple colored console formatter."""

    COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
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
    ):
        """Log an exception message at the specified level and then re-raise it.
        Args:
            exc: The exception to log and raise.
            message: Optional custom message, to override the exception's in the logs.
            level: The logging level (e.g., "error", "warning").
            include_traceback: Whether to include the traceback in the log.
        """
        log_message = message or str(exc)

        try:
            log_fn = getattr(self, level.lower(), None)
        except Exception as e:
            # If the log level is invalid, log the error and raise a ConfigError
            self.error(f"Invalid log level: {level}. Error: {e}", exc_info=True)
            raise ConfigError(f"Invalid log level: {level}.") from e

        log_fn(log_message, exc_info=include_traceback)  # log the exception message
        raise exc  # Reraise the exception after logging


def configure_logging(
    *,
    level: str = "INFO",
    log_dir: Path | None = None,
    force: bool = False,
) -> None:
    """
    Configure application logging.

    Creates:
    - console logger
    - JSON-lines file logger

    File structure:
        logs/YYYY/MM/DD/app.log
    """

    global _logging_configured

    if _logging_configured and not force:
        return

    # Register custom logger class
    logging.setLoggerClass(EnhancedLogger)

    resolved_level = level.upper()
    resolved_log_dir = log_dir or _get_log_dir()

    # Root logger
    root = logging.getLogger()

    root.handlers.clear()
    root.setLevel(resolved_level)

    # -------------------------------------------------------------------------
    # Console handler
    # -------------------------------------------------------------------------

    console_handler = logging.StreamHandler()

    console_handler.setLevel(resolved_level)
    console_handler.setFormatter(_ConsoleFormatter())

    root.addHandler(console_handler)

    # -------------------------------------------------------------------------
    # File handler
    # -------------------------------------------------------------------------

    log_file = _build_daily_log_path(resolved_log_dir)

    file_handler = logging.FileHandler(
        filename=log_file,
        encoding="utf-8",
    )

    file_handler.setLevel(resolved_level)
    file_handler.setFormatter(_JsonLinesFormatter())

    root.addHandler(file_handler)

    _logging_configured = True


def get_logger(name: str | None = None) -> EnhancedLogger:
    """
    Returns a module logger and ensures logging is configured once.

    Usage:
        logger = get_logger()
        logger.info("Hello world")
    """

    configure_logging()
    if name:
        return logging.getLogger(name)

    frame = inspect.currentframe()
    if frame is None or frame.f_back is None:
        return logging.getLogger(__name__)

    caller_globals = frame.f_back.f_globals
    caller_name = caller_globals.get("__name__", __name__)
    return logging.getLogger(caller_name)
