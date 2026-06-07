"""Custom logger type used across the application."""

from __future__ import annotations

import logging
from typing import NoReturn

from chess_teacher.utils.exception_utils import ConfigError


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
            self.error(f"Invalid log level: {level}. Error: {e}", exc_info=True)
            raise ConfigError(f"Invalid log level: {level}.") from e

        log_fn(log_message, exc_info=include_traceback)
        raise exc
