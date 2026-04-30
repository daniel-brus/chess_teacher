import inspect
import json
import logging
import os
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(
    *,
    level: str | None = None,
    json_logs: bool | None = None,
    log_file: str | None = None,
    max_bytes: int | None = None,
    backup_count: int | None = None,
    force: bool = False,
) -> None:
    """
    Basic logging for local + Docker.

    Env vars:
    - LOG_LEVEL (default INFO)
    - LOG_JSON=1 for JSON lines
    - LOG_FILE=/path/to/app.log for rotating file logs
    - LOG_MAX_BYTES (default 5_000_000)
    - LOG_BACKUP_COUNT (default 3)
    """

    root = logging.getLogger()
    if getattr(root, "_chess_teacher_logging_configured", False) and not force:
        return

    resolved_level = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    resolved_json = json_logs if json_logs is not None else _env_bool("LOG_JSON", False)
    resolved_log_file = log_file if log_file is not None else os.getenv("LOG_FILE")
    resolved_max_bytes = (
        max_bytes if max_bytes is not None else _env_int("LOG_MAX_BYTES", 5_000_000)
    )
    resolved_backup_count = (
        backup_count if backup_count is not None else _env_int("LOG_BACKUP_COUNT", 3)
    )

    root.handlers.clear()
    root.setLevel(resolved_level)

    if resolved_json:
        formatter: logging.Formatter = _JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(resolved_level)
    root.addHandler(console)

    if resolved_log_file:
        log_path = Path(resolved_log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=resolved_max_bytes,
            backupCount=resolved_backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(resolved_level)
        root.addHandler(file_handler)

    root._chess_teacher_logging_configured = True


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Returns a module logger and ensures logging is configured once.

    Usage:
        logger = get_logger()
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
