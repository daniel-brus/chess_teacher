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

# The wait runs after this pipeline is already resident, so MemAvailable is
# the budget for the TensorFlow child. Loading the hybrid model drops that
# budget by a few hundred MB. Prod fits that started with about 950 MB still
# free were OOM-killed on the 4 GB node (Nov 2026). 1.4 GB before the child
# starts leaves about 1 GB after the weights load.
DEFAULT_MIN_PIPELINE_MEM_AVAILABLE_MB = 1400.0
MIN_PIPELINE_MEM_AVAILABLE_MB_ENV = "PIPELINE_MIN_MEM_AVAILABLE_MB"
# Checked again after the weights are in memory, before Keras allocates the
# training step. The killed runs were under 1000 MB at that point.
DEFAULT_MIN_FIT_MEM_AVAILABLE_MB = 1000.0
MIN_FIT_MEM_AVAILABLE_MB_ENV = "PIPELINE_MIN_FIT_MEM_AVAILABLE_MB"


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


class FitMemoryError(RuntimeError):
    """Starting or continuing a Keras fit would OOM-kill the host."""


def _positive_mb(env_name: str, default: float) -> float:
    raw = get_optional_env_variable(env_name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %.0f.", env_name, raw, default)
        return default
    if value <= 0:
        logger.warning("%s=%s is below 1; using %.0f.", env_name, value, default)
        return default
    return value


def min_pipeline_mem_available_mb() -> float:
    """Minimum host MemAvailable before the TensorFlow child may start."""
    return _positive_mb(
        MIN_PIPELINE_MEM_AVAILABLE_MB_ENV,
        DEFAULT_MIN_PIPELINE_MEM_AVAILABLE_MB,
    )


def min_fit_mem_available_mb() -> float:
    """Minimum host MemAvailable after the model is loaded, before a fit step."""
    return _positive_mb(MIN_FIT_MEM_AVAILABLE_MB_ENV, DEFAULT_MIN_FIT_MEM_AVAILABLE_MB)


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


def fit_host_has_memory() -> bool:
    """True when MemAvailable can take another hybrid fit step.

    An unreadable MemAvailable does not block the fit.
    """
    available = snapshot_host_pressure().mem_available_mb
    minimum = min_fit_mem_available_mb()
    if available is None:
        logger.warning("MemAvailable unreadable; allowing Keras fit.")
        return True
    ok = available >= minimum
    logger.info(
        "Fit memory check mem_available_mb=%.0f minimum_mb=%.0f ok=%s",
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
