"""Shared partial-row DB checkpoint helpers for long FEN-evaluation transforms."""

from __future__ import annotations

from chess_teacher.utils.db.client import MergeStrategy
from chess_teacher.utils.env_utils import get_optional_env_variable
from chess_teacher.utils.process_utils import WorkerSafeLogger

_logger = WorkerSafeLogger(__name__)

CHECKPOINT_MERGE = MergeStrategy(
    when_matched="update",
    when_not_matched_by_target="ignore",
    when_not_matched_by_source="ignore",
)

_DEFAULT_CHECKPOINT_PERCENT = 10


def checkpoint_percent_from_env(env_var: str, *, default: int = _DEFAULT_CHECKPOINT_PERCENT) -> int | None:
    """Read a checkpoint interval (percent) from env; ``0`` or invalid disables checkpoints."""
    raw = get_optional_env_variable(env_var)
    if raw is None:
        return default
    if not str(raw).strip():
        return None
    try:
        value = int(raw)
    except ValueError:
        _logger.warning("Invalid %s=%r; using default %s", env_var, raw, default)
        return default
    if value <= 0:
        return None
    return value
