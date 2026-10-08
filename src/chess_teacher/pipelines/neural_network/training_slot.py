"""One global slot for neural training, plus the host-memory check."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from chess_teacher.utils.db.client import DatabaseClient
from chess_teacher.utils.env_utils import get_optional_env_variable
from chess_teacher.utils.general_utils import (
    as_utc,
    generate_ident_is_literal,
    quote_ident,
    quote_literal,
)
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.process_utils import snapshot_host_pressure
from chess_teacher.utils.table_data_class import TableDataClass

logger = get_logger()

GLOBAL_TRAINING_SLOT = "global"
# A fit is well under an hour. Three hours lets a killed holder go stale
# without handing the slot to a second fit that is still running.
TRAINING_SLOT_STALE_AFTER = timedelta(hours=3)
# Stop waiting before the 6-hour per-user pipeline lock expires.
TRAINING_SLOT_WAIT_LIMIT = timedelta(hours=4)
TRAINING_SLOT_POLL_SECONDS = 30.0

# One trainer on the 4 GB node still had about 1.1 GB free when it finished.
# Overlap was already fatal around that level, so the next fit waits until
# about 1.5 GB is free.
DEFAULT_MIN_PIPELINE_MEM_AVAILABLE_MB = 1500.0
MIN_PIPELINE_MEM_AVAILABLE_MB_ENV = "PIPELINE_MIN_MEM_AVAILABLE_MB"


@dataclass(frozen=True)
class TrainingSlotRecord(TableDataClass):
    """Single-row lock. ``holder_run_id`` empty means the slot is free."""

    slot_name: str
    holder_run_id: str | None = None
    acquired_at: datetime | None = None

    @classmethod
    def get_yaml_path(cls) -> Path:
        return Path(__file__).parent / "metadata.yml"

    @classmethod
    def get_key(cls) -> str:
        return "training_slot"

    @classmethod
    def get_id_hash_columns(cls) -> tuple[str, ...]:
        return ()


def min_pipeline_mem_available_mb() -> float:
    """Minimum host MemAvailable before a training fit may start."""
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
    """True when MemAvailable can hold another training fit.

    An unreadable MemAvailable does not block the fit.
    """
    available = snapshot_host_pressure().mem_available_mb
    minimum = min_pipeline_mem_available_mb()
    if available is None:
        logger.warning("MemAvailable unreadable; allowing a training fit.")
        return True
    ok = available >= minimum
    logger.info(
        "Training memory check mem_available_mb=%.0f minimum_mb=%.0f ok=%s",
        available,
        minimum,
        ok,
    )
    return ok


class DbTrainingSlot:
    """Postgres-backed global training slot."""

    def __init__(self, db_client: DatabaseClient) -> None:
        self._db = db_client

    def try_acquire(self, holder_run_id: str, *, now: datetime) -> bool:
        """Take the slot when it is free, already ours, or older than the stale window."""
        self._ensure_row()
        stale_before = as_utc(now) - TRAINING_SLOT_STALE_AFTER
        stale_clause = (
            f"{quote_ident('acquired_at')} < {quote_literal(stale_before.isoformat())}::timestamptz"
        )
        where = " AND ".join([
            generate_ident_is_literal("slot_name", GLOBAL_TRAINING_SLOT),
            "("
            + " OR ".join([
                generate_ident_is_literal("holder_run_id", None),
                generate_ident_is_literal("holder_run_id", holder_run_id),
                stale_clause,
            ])
            + ")",
        ])
        affected = self._db.update_where(
            TrainingSlotRecord.get_metadata(),
            {"holder_run_id": holder_run_id, "acquired_at": as_utc(now)},
            where,
        )
        return affected == 1

    def release(self, holder_run_id: str) -> None:
        """Drop the slot only when this run still holds it."""
        where = " AND ".join([
            generate_ident_is_literal("slot_name", GLOBAL_TRAINING_SLOT),
            generate_ident_is_literal("holder_run_id", holder_run_id),
        ])
        self._db.update_where(
            TrainingSlotRecord.get_metadata(),
            {"holder_run_id": None, "acquired_at": None},
            where,
        )

    def _ensure_row(self) -> None:
        TrainingSlotRecord(slot_name=GLOBAL_TRAINING_SLOT).save_new_to_db(self._db)
