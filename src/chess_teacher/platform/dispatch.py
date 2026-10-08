"""Choose which user pipeline the 30-minute dispatcher may start."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from chess_teacher.platform.user import User
from chess_teacher.utils.env_utils import get_optional_env_variable
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.process_utils import snapshot_host_pressure

logger = get_logger()

# One trainer on the 4 GB node still had about 1.1 GB free when it finished.
# Overlap was already fatal around that level, so the next job waits until
# about 1.5 GB is free.
DEFAULT_MIN_PIPELINE_MEM_AVAILABLE_MB = 1500.0
MIN_PIPELINE_MEM_AVAILABLE_MB_ENV = "PIPELINE_MIN_MEM_AVAILABLE_MB"

DEFER_MEMORY = "memory"
DEFER_ACTIVE_JOB = "active_job"
DEFER_WAITING_TURN = "waiting_turn"


@dataclass(frozen=True)
class PipelineDispatchPlan:
    """One dispatcher tick: at most one spawn, and who waits for the next tick."""

    spawn_user_id: str | None
    defer_user_ids: tuple[str, ...]
    defer_reason: str | None
    skipped_not_due: int
    skipped_cooldown: int
    skipped_active_job: int


def min_pipeline_mem_available_mb() -> float:
    """Minimum host MemAvailable before another pipeline job may start."""
    raw = get_optional_env_variable(MIN_PIPELINE_MEM_AVAILABLE_MB_ENV)
    if not raw:
        return DEFAULT_MIN_PIPELINE_MEM_AVAILABLE_MB
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not a number; using %.0f.",
            MIN_PIPELINE_MEM_AVAILABLE_MB_ENV,
            raw,
            DEFAULT_MIN_PIPELINE_MEM_AVAILABLE_MB,
        )
        return DEFAULT_MIN_PIPELINE_MEM_AVAILABLE_MB
    if value <= 0:
        logger.warning(
            "%s=%s is below 1; using %.0f.",
            MIN_PIPELINE_MEM_AVAILABLE_MB_ENV,
            value,
            DEFAULT_MIN_PIPELINE_MEM_AVAILABLE_MB,
        )
        return DEFAULT_MIN_PIPELINE_MEM_AVAILABLE_MB
    return value


def pipeline_host_has_memory() -> bool:
    """True when MemAvailable can hold another training run.

    An unreadable MemAvailable does not block the start.
    """
    available = snapshot_host_pressure().mem_available_mb
    minimum = min_pipeline_mem_available_mb()
    if available is None:
        logger.warning("MemAvailable unreadable; allowing a pipeline start.")
        return True
    ok = available >= minimum
    logger.info(
        "Pipeline memory check mem_available_mb=%.0f minimum_mb=%.0f ok=%s",
        available,
        minimum,
        ok,
    )
    return ok


def _cron_instant_utc(user: User, now: datetime) -> datetime:
    local_now = now.astimezone(ZoneInfo(user.timezone))
    scheduled = local_now.replace(
        hour=user.cron_time.hour,
        minute=user.cron_time.minute,
        second=0,
        microsecond=0,
    )
    return scheduled.astimezone(UTC)


def plan_pipeline_dispatch(
    users: Sequence[User],
    *,
    now: datetime,
    active_user_ids: set[str],
    pipeline_job_active: bool,
    cooldown_blocked_user_ids: set[str],
    memory_available: bool,
) -> PipelineDispatchPlan:
    """Pick at most one user to start, and queue the rest for the next tick.

    A user is eligible in the cron window, or whenever
    ``run_pipeline_immediately`` is set. While any pipeline job is still
    running, or MemAvailable is under the minimum, nobody new starts.
    """
    skipped_not_due = 0
    skipped_cooldown = 0
    skipped_active = 0
    eligible: list[User] = []

    for user in users:
        if not user.is_pipeline_dispatch_due(now):
            skipped_not_due += 1
            continue
        if user.user_id in cooldown_blocked_user_ids:
            skipped_cooldown += 1
            continue
        if user.user_id in active_user_ids:
            skipped_active += 1
            continue
        eligible.append(user)

    eligible.sort(key=lambda user: (_cron_instant_utc(user, now), user.user_id))
    eligible_ids = tuple(user.user_id for user in eligible)

    if not eligible_ids:
        return PipelineDispatchPlan(
            spawn_user_id=None,
            defer_user_ids=(),
            defer_reason=None,
            skipped_not_due=skipped_not_due,
            skipped_cooldown=skipped_cooldown,
            skipped_active_job=skipped_active,
        )

    if pipeline_job_active:
        return PipelineDispatchPlan(
            spawn_user_id=None,
            defer_user_ids=eligible_ids,
            defer_reason=DEFER_ACTIVE_JOB,
            skipped_not_due=skipped_not_due,
            skipped_cooldown=skipped_cooldown,
            skipped_active_job=skipped_active,
        )

    if not memory_available:
        return PipelineDispatchPlan(
            spawn_user_id=None,
            defer_user_ids=eligible_ids,
            defer_reason=DEFER_MEMORY,
            skipped_not_due=skipped_not_due,
            skipped_cooldown=skipped_cooldown,
            skipped_active_job=skipped_active,
        )

    return PipelineDispatchPlan(
        spawn_user_id=eligible_ids[0],
        defer_user_ids=eligible_ids[1:],
        defer_reason=DEFER_WAITING_TURN if len(eligible_ids) > 1 else None,
        skipped_not_due=skipped_not_due,
        skipped_cooldown=skipped_cooldown,
        skipped_active_job=skipped_active,
    )
