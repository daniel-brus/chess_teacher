"""Console and JSON-lines formatters for application logs."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any, ClassVar

from chess_teacher.utils.env_utils import get_env_variable


class JsonLinesFormatter(logging.Formatter):
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


class ConsoleFormatter(logging.Formatter):
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
