"""Dispatcher queue: one pipeline at a time, retried on the next 30-minute tick."""

from __future__ import annotations

from datetime import UTC, datetime, time

import pytest

from chess_teacher.platform.dispatch import (
    DEFER_ACTIVE_JOB,
    DEFER_MEMORY,
    DEFER_WAITING_TURN,
    min_pipeline_mem_available_mb,
    pipeline_host_has_memory,
    plan_pipeline_dispatch,
)
from chess_teacher.platform.user import User
from chess_teacher.utils.process_utils import HostPressure

# 2026-10-08 is still CEST (UTC+2).
_AMSTERDAM_0305 = datetime(2026, 10, 8, 1, 5, tzinfo=UTC)
_AMSTERDAM_0310 = datetime(2026, 10, 8, 1, 10, tzinfo=UTC)
_AMSTERDAM_0340 = datetime(2026, 10, 8, 1, 40, tzinfo=UTC)
_AMSTERDAM_0210_NEXT = datetime(2026, 10, 9, 0, 10, tzinfo=UTC)


def _user(
    user_id: str,
    *,
    cron: time = time(3, 0),
    retry_at: datetime | None = None,
    timezone: str = "Europe/Amsterdam",
) -> User:
    return User(
        user_id=user_id,
        sub=user_id,
        provider="test",
        cron_time=cron,
        timezone=timezone,
        pipeline_retry_at=retry_at,
    )


def _plan(
    users: list[User],
    now: datetime,
    *,
    active_user_ids: set[str] | None = None,
    pipeline_job_active: bool = False,
    cooldown_blocked_user_ids: set[str] | None = None,
    memory_available: bool = True,
):
    return plan_pipeline_dispatch(
        users,
        now=now,
        active_user_ids=active_user_ids or set(),
        pipeline_job_active=pipeline_job_active,
        cooldown_blocked_user_ids=cooldown_blocked_user_ids or set(),
        memory_available=memory_available,
    )


def test_users_metadata_matches_dataclass() -> None:
    User.assert_metadata_sync()


def test_cron_window_starts_the_only_due_user() -> None:
    plan = _plan([_user("anna")], _AMSTERDAM_0310)
    assert plan.spawn_user_id == "anna"
    assert plan.defer_user_ids == ()
    assert plan.defer_reason is None


def test_same_slot_starts_one_user_and_queues_the_rest() -> None:
    plan = _plan([_user("jonathan"), _user("anna")], _AMSTERDAM_0310)
    assert plan.spawn_user_id == "anna"
    assert plan.defer_user_ids == ("jonathan",)
    assert plan.defer_reason == DEFER_WAITING_TURN


def test_queued_user_is_due_after_the_cron_window() -> None:
    user = _user("jonathan", retry_at=_AMSTERDAM_0305)
    assert user.is_cron_due(_AMSTERDAM_0340) is False
    assert user.is_pipeline_dispatch_due(_AMSTERDAM_0340) is True
    plan = _plan([user], _AMSTERDAM_0340)
    assert plan.spawn_user_id == "jonathan"


def test_user_without_a_retry_is_not_due_after_the_window() -> None:
    user = _user("jonathan")
    assert user.is_pipeline_dispatch_due(_AMSTERDAM_0340) is False
    plan = _plan([user], _AMSTERDAM_0340)
    assert plan.spawn_user_id is None
    assert plan.defer_user_ids == ()
    assert plan.skipped_not_due == 1


def test_stale_retry_does_not_start_before_the_next_cron() -> None:
    user = _user("jonathan", retry_at=_AMSTERDAM_0305)
    assert user.is_pipeline_dispatch_due(_AMSTERDAM_0210_NEXT) is False


def test_low_memory_queues_every_due_user() -> None:
    plan = _plan(
        [_user("anna"), _user("jonathan")],
        _AMSTERDAM_0310,
        memory_available=False,
    )
    assert plan.spawn_user_id is None
    assert plan.defer_user_ids == ("anna", "jonathan")
    assert plan.defer_reason == DEFER_MEMORY


def test_an_active_pipeline_job_queues_the_other_due_user() -> None:
    plan = _plan(
        [_user("anna"), _user("jonathan")],
        _AMSTERDAM_0310,
        active_user_ids={"anna"},
        pipeline_job_active=True,
    )
    assert plan.spawn_user_id is None
    assert plan.skipped_active_job == 1
    assert plan.defer_user_ids == ("jonathan",)
    assert plan.defer_reason == DEFER_ACTIVE_JOB


def test_cooldown_does_not_queue_the_user() -> None:
    plan = _plan(
        [_user("anna")],
        _AMSTERDAM_0310,
        cooldown_blocked_user_ids={"anna"},
    )
    assert plan.spawn_user_id is None
    assert plan.defer_user_ids == ()
    assert plan.skipped_cooldown == 1


def test_earlier_cron_runs_before_a_later_one_once_both_are_queued() -> None:
    plan = _plan(
        [
            _user("later", cron=time(3, 30), retry_at=_AMSTERDAM_0305),
            _user("earlier", cron=time(3, 0), retry_at=_AMSTERDAM_0305),
        ],
        _AMSTERDAM_0340,
    )
    assert plan.spawn_user_id == "earlier"
    assert plan.defer_user_ids == ("later",)


def test_min_memory_falls_back_when_the_env_value_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PIPELINE_MIN_MEM_AVAILABLE_MB", "lots")
    assert min_pipeline_mem_available_mb() == 1500.0


def test_memory_check_uses_mem_available(monkeypatch: pytest.MonkeyPatch) -> None:
    def _pressure() -> HostPressure:
        return HostPressure(
            rss_mb=100.0,
            cpu_count=2,
            affinity_count=2,
            load1=0.2,
            mem_available_mb=1100.0,
            mem_total_mb=4096.0,
        )

    monkeypatch.setattr(
        "chess_teacher.platform.dispatch.snapshot_host_pressure",
        _pressure,
    )
    assert pipeline_host_has_memory() is False

    def _enough() -> HostPressure:
        return HostPressure(
            rss_mb=100.0,
            cpu_count=2,
            affinity_count=2,
            load1=0.2,
            mem_available_mb=1800.0,
            mem_total_mb=4096.0,
        )

    monkeypatch.setattr(
        "chess_teacher.platform.dispatch.snapshot_host_pressure",
        _enough,
    )
    assert pipeline_host_has_memory() is True
