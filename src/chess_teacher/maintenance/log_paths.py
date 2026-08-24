"""Path helpers for closed log segment keys in object storage."""

from __future__ import annotations

from datetime import date
from pathlib import PurePosixPath

CLOSED_LOG_STORAGE_PREFIX = "logs/python/buffer/closed"
PROCESSED_LOG_STORAGE_PREFIX = "logs/python/buffer/processed"
UNKNOWN_HOSTNAME = "unknown"


def relative_closed_log_key(
    source_file: str,
    *,
    prefix: str = CLOSED_LOG_STORAGE_PREFIX,
) -> str:
    """Return the path under ``prefix`` for a storage key."""
    full_prefix = f"{prefix}/"
    if source_file.startswith(full_prefix):
        return source_file[len(full_prefix) :]
    return source_file


def parse_closed_log_path_date(relative_key: str) -> date | None:
    """
    Parse the log segment date from a key relative to ``CLOSED_LOG_STORAGE_PREFIX``.

    Expected layout: ``{YYYY}/{MM}/{DD}/{hostname}/{segment}.log``
    """
    parts = PurePosixPath(relative_key).parts
    if len(parts) < 4:
        return None
    try:
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return None


def parse_closed_log_hostname(
    source_file: str,
    *,
    prefix: str = CLOSED_LOG_STORAGE_PREFIX,
) -> str:
    """
    Parse hostname/pod from a closed log segment storage key.

    Expected layout: ``.../closed/{YYYY}/{MM}/{DD}/{hostname}/{segment}.log``
    """
    relative = relative_closed_log_key(source_file, prefix=prefix)
    if parse_closed_log_path_date(relative) is None:
        return UNKNOWN_HOSTNAME
    parts = PurePosixPath(relative).parts
    if len(parts) >= 4:
        return parts[3]
    return UNKNOWN_HOSTNAME
