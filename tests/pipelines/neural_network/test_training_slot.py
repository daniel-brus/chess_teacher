"""Global training slot: wait for the lock and for free memory, then release on finish."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from chess_teacher.pipelines.neural_network.scheme_steps import (
    SKIP_KEY,
    WaitForTrainingSlotStep,
)
from chess_teacher.pipelines.neural_network.training_slot import (
    DbTrainingSlot,
    TrainingSlotRecord,
    min_pipeline_mem_available_mb,
    pipeline_host_has_memory,
)
from chess_teacher.utils.pipeline_utils.pipeline_base import (
    PIPELINE_CLEANUPS_EXTRA,
    PIPELINE_RUN_ID_EXTRA,
    PipelineContext,
)
from chess_teacher.utils.process_utils import HostPressure


class _Scheme:
    min_new_moves = 10

    def __init__(self, pending: int = 25) -> None:
        self.pending = pending
        self.checked = 0

    def note_checked(self, db_client: object) -> None:
        del db_client
        self.checked += 1

    def count_pending(self, db_client: object) -> int:
        del db_client
        return self.pending


class _Slot:
    def __init__(self, answers: list[bool] | None = None) -> None:
        self.answers = answers
        self.holder: str | None = None
        self.acquire_calls = 0
        self.release_calls = 0

    def try_acquire(self, holder_run_id: str, *, now: datetime) -> bool:
        del now
        self.acquire_calls += 1
        ok = True if self.answers is None else self.answers.pop(0)
        if ok:
            self.holder = holder_run_id
        return ok

    def release(self, holder_run_id: str) -> None:
        self.release_calls += 1
        if self.holder == holder_run_id:
            self.holder = None


def _context() -> PipelineContext:
    context = PipelineContext()
    context.extras[PIPELINE_RUN_ID_EXTRA] = "run-1"
    return context


def _step(
    scheme: _Scheme,
    slot: _Slot,
    *,
    memory: list[bool],
    clock: dict[str, float],
    sleeps: list[float],
    wait_seconds: float = 60,
    poll_seconds: float = 30,
) -> WaitForTrainingSlotStep:
    def _memory() -> bool:
        return memory.pop(0)

    def _sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock["t"] += seconds

    return WaitForTrainingSlotStep(
        scheme,  # type: ignore[arg-type]
        slot=slot,  # type: ignore[arg-type]
        memory_available=_memory,
        sleep=_sleep,
        monotonic=lambda: clock["t"],
        poll_seconds=poll_seconds,
        wait_seconds=wait_seconds,
    )


def test_training_slot_metadata_matches_dataclass() -> None:
    TrainingSlotRecord.assert_metadata_sync()


def test_low_pending_skips_without_taking_the_slot() -> None:
    scheme = _Scheme(pending=3)
    slot = _Slot()
    context = _context()
    step = _step(scheme, slot, memory=[True], clock={"t": 0}, sleeps=[])
    step.run(MagicMock(), context)  # type: ignore[arg-type]
    assert context.extras[SKIP_KEY] is True
    assert scheme.checked == 1
    assert slot.acquire_calls == 0


def test_free_slot_and_memory_acquire_and_register_release() -> None:
    scheme = _Scheme()
    slot = _Slot()
    context = _context()
    sleeps: list[float] = []
    step = _step(scheme, slot, memory=[True], clock={"t": 0}, sleeps=sleeps)
    step.run(MagicMock(), context)  # type: ignore[arg-type]
    assert sleeps == []
    assert slot.holder == "run-1"
    assert SKIP_KEY not in context.extras
    cleanups = context.extras[PIPELINE_CLEANUPS_EXTRA]
    cleanups[0]()
    assert slot.holder is None
    assert slot.release_calls == 1


def test_low_memory_releases_the_slot_and_tries_again() -> None:
    scheme = _Scheme()
    slot = _Slot()
    context = _context()
    sleeps: list[float] = []
    step = _step(
        scheme,
        slot,
        memory=[False, True],
        clock={"t": 0.0},
        sleeps=sleeps,
    )
    step.run(MagicMock(), context)  # type: ignore[arg-type]
    assert sleeps == [30]
    assert slot.holder == "run-1"
    assert slot.release_calls == 1


def test_busy_slot_waits_until_it_is_free() -> None:
    scheme = _Scheme()
    slot = _Slot(answers=[False, True])
    context = _context()
    sleeps: list[float] = []
    step = _step(scheme, slot, memory=[True], clock={"t": 0.0}, sleeps=sleeps)
    step.run(MagicMock(), context)  # type: ignore[arg-type]
    assert sleeps == [30]
    assert slot.holder == "run-1"
    assert slot.acquire_calls == 2


def test_wait_cap_skips_the_round_and_does_not_hold_the_slot() -> None:
    scheme = _Scheme()
    slot = _Slot()
    context = _context()
    sleeps: list[float] = []
    step = _step(
        scheme,
        slot,
        memory=[False, False, False],
        clock={"t": 0.0},
        sleeps=sleeps,
        wait_seconds=60,
        poll_seconds=30,
    )
    step.run(MagicMock(), context)  # type: ignore[arg-type]
    assert sleeps == [30, 30]
    assert scheme.checked == 1
    assert context.extras[SKIP_KEY] is True
    assert slot.holder is None
    assert PIPELINE_CLEANUPS_EXTRA not in context.extras


def test_try_acquire_requires_a_free_or_stale_row() -> None:
    db = MagicMock()
    db.update_where.return_value = 1
    now = datetime(2026, 10, 8, 16, 0, tzinfo=UTC)
    assert DbTrainingSlot(db).try_acquire("run-1", now=now) is True
    where = db.update_where.call_args.args[2]
    assert '"holder_run_id" IS NULL' in where
    assert "\"holder_run_id\" = 'run-1'" in where
    assert "::timestamptz" in where

    db.update_where.return_value = 0
    assert DbTrainingSlot(db).try_acquire("run-1", now=now) is False


def test_min_memory_falls_back_when_the_env_value_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PIPELINE_MIN_MEM_AVAILABLE_MB", "lots")
    assert min_pipeline_mem_available_mb() == 1500.0


def test_memory_check_uses_mem_available(monkeypatch: pytest.MonkeyPatch) -> None:
    def _low() -> HostPressure:
        return HostPressure(
            rss_mb=100.0,
            cpu_count=2,
            affinity_count=2,
            load1=0.2,
            mem_available_mb=1100.0,
            mem_total_mb=4096.0,
        )

    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.training_slot.snapshot_host_pressure", _low
    )
    assert pipeline_host_has_memory() is False
