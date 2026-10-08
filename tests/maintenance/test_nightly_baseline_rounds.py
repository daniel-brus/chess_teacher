"""Nightly baseline train-and-promote repeats only while new train moves remain."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from chess_teacher.maintenance import main as nightly
from chess_teacher.pipelines.neural_network.schemes import ModelTraining
from chess_teacher.utils.pipeline_utils.pipeline_helpers import (
    PipelineResult,
    PipelineRunResult,
)


def _result(result: PipelineResult) -> PipelineRunResult:
    now = datetime.now(UTC)
    return PipelineRunResult(
        run_id="run-test",
        name="baseline_training",
        user_id=None,
        account_id=None,
        result=result,
        started_at=now,
        finished_at=now,
    )


def test_resolve_baseline_nightly_rounds_defaults_to_one() -> None:
    assert nightly.resolve_baseline_nightly_rounds("") == 1
    assert nightly.resolve_baseline_nightly_rounds("  ") == 1
    assert nightly.resolve_baseline_nightly_rounds("3") == 3
    assert nightly.resolve_baseline_nightly_rounds("0") == 1
    assert nightly.resolve_baseline_nightly_rounds("nope") == 1


def test_first_round_always_runs_and_later_rounds_stop_when_pending_is_low(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    def _train(*, promote: bool = False) -> PipelineRunResult:
        calls.append(promote)
        return _result(PipelineResult.SUCCESS)

    monkeypatch.setattr(nightly, "run_baseline_training_pipeline", _train)
    monkeypatch.setattr(nightly, "get_db_client", lambda: object())
    monkeypatch.setattr(ModelTraining, "count_pending", lambda self, db: 0)

    results = nightly.run_nightly_baseline_rounds(3)
    assert len(results) == 1
    assert calls == [True]


def test_extra_rounds_run_while_pending_stays_above_the_minimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0}

    def _train(*, promote: bool = False) -> PipelineRunResult:
        calls["n"] += 1
        return _result(PipelineResult.SUCCESS)

    monkeypatch.setattr(nightly, "run_baseline_training_pipeline", _train)
    monkeypatch.setattr(nightly, "get_db_client", lambda: object())
    monkeypatch.setattr(ModelTraining, "count_pending", lambda self, db: 5000)

    results = nightly.run_nightly_baseline_rounds(3)
    assert len(results) == 3
    assert calls["n"] == 3


def test_failed_round_stops_the_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def _train(*, promote: bool = False) -> PipelineRunResult:
        calls["n"] += 1
        if calls["n"] == 2:
            return _result(PipelineResult.FAILURE)
        return _result(PipelineResult.SUCCESS)

    monkeypatch.setattr(nightly, "run_baseline_training_pipeline", _train)
    monkeypatch.setattr(nightly, "get_db_client", lambda: object())
    monkeypatch.setattr(ModelTraining, "count_pending", lambda self, db: 5000)

    results = nightly.run_nightly_baseline_rounds(4)
    assert [result.result for result in results] == [
        PipelineResult.SUCCESS,
        PipelineResult.FAILURE,
    ]
