"""Tests for MAX_ACCOUNT_WORKERS resolution on PipelineRunner."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from chess_teacher.pipelines.runner import (
    _DEFAULT_MAX_ACCOUNT_WORKERS,
    PipelineRunner,
    resolve_max_account_workers,
)


@pytest.mark.parametrize(
    ("explicit", "env_value", "expected"),
    [
        (None, "", 2),
        (None, "   ", 2),
        (None, "1", 1),
        (None, "4", 4),
        (None, "0", 1),  # clamped
        (3, "1", 3),  # explicit wins
        (0, "", 1),  # clamped
    ],
)
def test_resolve_max_account_workers(
    explicit: int | None,
    env_value: str,
    expected: int,
) -> None:
    assert resolve_max_account_workers(explicit=explicit, env_value=env_value) == expected


def test_resolve_max_account_workers_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_ACCOUNT_WORKERS", "3")
    assert resolve_max_account_workers() == 3


def test_resolve_max_account_workers_unset_env_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MAX_ACCOUNT_WORKERS", raising=False)
    assert resolve_max_account_workers() == 2


def test_resolve_max_account_workers_default_constant() -> None:
    assert _DEFAULT_MAX_ACCOUNT_WORKERS == 2


def test_pipeline_runner_uses_env_when_explicit_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_ACCOUNT_WORKERS", "1")
    runner = PipelineRunner(user=MagicMock(), db_client=MagicMock())
    assert runner.max_account_workers == 1


def test_pipeline_runner_explicit_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_ACCOUNT_WORKERS", "1")
    runner = PipelineRunner(
        user=MagicMock(),
        db_client=MagicMock(),
        max_account_workers=4,
    )
    assert runner.max_account_workers == 4


def test_resolve_max_account_workers_invalid_env_raises() -> None:
    with pytest.raises(ValueError):
        resolve_max_account_workers(env_value="not-an-int")
