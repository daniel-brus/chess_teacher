import inspect
import json
import logging
import os
from datetime import UTC, datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, ClassVar


def _get_log_dir() -> Path:
    """Get the log directory path from env or default."""
    base = os.getenv("LOG_DIR")
    return Path(base)


class _JsonLinesFormatter(logging.Formatter):
    """Minimal JSON lines formatter."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Exception info
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

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
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        return (
            f"{timestamp} {color}{record.levelname:<8}{self.RESET} "
            f"{record.name}: {record.getMessage()}"
        )


def configure_logging(
    *,
    level: str | None = None,
    log_dir: Path | None = None,
    force: bool = False,
) -> None:
    """
    Configure logging with JSON lines file + console output.

    Env vars:
    - LOG_LEVEL (default INFO)
    - LOG_DIR (default storage/logs)
    - ENVIRONMENT (default development)
    """

    root = logging.getLogger()
    if getattr(root, "_chess_teacher_logging_configured", False) and not force:
        return

    resolved_level = (level or "INFO").upper()
    resolved_log_dir = log_dir or _get_log_dir()

    root.handlers.clear()
    root.setLevel(resolved_level)

    # Console handler with colored output
    console = logging.StreamHandler()
    console.setFormatter(_ConsoleFormatter())
    console.setLevel(resolved_level)
    root.addHandler(console)

    # File handler with daily JSON lines
    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    log_file = resolved_log_dir / "app.log"

    file_handler = TimedRotatingFileHandler(
        log_file,
        when="midnight",
        interval=1,
        backupCount=30,  # Keep 30 days of logs
        encoding="utf-8",
    )
    file_handler.suffix = "%Y/%m/%d"  # Directory structure: YYYY/MM/DD/app.log
    file_handler.setFormatter(_JsonLinesFormatter())
    file_handler.setLevel(resolved_level)
    root.addHandler(file_handler)

    root._chess_teacher_logging_configured = True


def get_logger(name: str | None = None) -> logging.Logger:
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
