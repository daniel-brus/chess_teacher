"""Unit tests for baseline catch-up loop (mocked pipelines / eligible counts)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from chess_teacher.pipelines.neural_network import catch_up
from chess_teacher.pipelines.neural_network.pipeline_steps import MIN_NEW_MOVES_BASELINE
from chess_teacher.utils.pipeline_utils.pipeline_helpers import (
    PipelineResult,
    PipelineRunResult,
    StepResult,
)


def _ts() -> datetime:
    return datetime.now(UTC)


def _ok_result(*, name: str = "baseline_training") -> PipelineRunResult:
    now = _ts()
    return PipelineRunResult(
        run_id="run-test",
        name=name,
        user_id=None,
        account_id=None,
        result=PipelineResult.SUCCESS,
        started_at=now,
        finished_at=now,
        step_results=(
            StepResult(
                name="CheckSufficientNewData",
                result=PipelineResult.SUCCESS,
                started_at=now,
                finished_at=now,
            ),
        ),
    )


def _fail_result(*, name: str = "baseline_training") -> PipelineRunResult:
    now = _ts()
    return PipelineRunResult(
        run_id="run-fail",
        name=name,
        user_id=None,
        account_id=None,
        result=PipelineResult.FAILURE,
        started_at=now,
        finished_at=now,
        step_results=(
            StepResult(
                name="TrainIncremental",
                result=PipelineResult.FAILURE,
                started_at=now,
                finished_at=now,
                error_message="boom",
            ),
        ),
    )


def test_run_ok_true_on_success() -> None:
    assert catch_up._run_ok(_ok_result(), label="train") is True


def test_run_ok_false_on_failure() -> None:
    assert catch_up._run_ok(_fail_result(), label="train") is False


def test_loop_already_caught_up_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        catch_up,
        "_eligible_count",
        lambda: (MIN_NEW_MOVES_BASELINE - 1, None),
    )
    train = MagicMock()
    promote = MagicMock()
    monkeypatch.setattr(catch_up, "run_baseline_training_pipeline", train)
    monkeypatch.setattr(catch_up, "run_baseline_promotion_pipeline", promote)

    assert catch_up.loop_until_caught_up(promote=True, max_rounds=5) == 0
    train.assert_not_called()
    promote.assert_not_called()


def test_loop_trains_and_promotes_until_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    # Round1: 2500 → train → 1500; Round2: 1500 → train → 500; then 500 → done.
    seq = iter([
        (2500, None),
        (1500, "c1"),
        (1500, "c1"),
        (500, "c2"),
        (500, "c2"),
    ])
    monkeypatch.setattr(catch_up, "_eligible_count", lambda: next(seq))
    train = MagicMock(return_value=_ok_result())
    promote = MagicMock(return_value=_ok_result(name="baseline_promotion"))
    monkeypatch.setattr(catch_up, "run_baseline_training_pipeline", train)
    monkeypatch.setattr(catch_up, "run_baseline_promotion_pipeline", promote)

    assert catch_up.loop_until_caught_up(promote=True, max_rounds=10) == 0
    assert train.call_count == 2
    assert promote.call_count == 2


def test_loop_no_promote_skips_promotion(monkeypatch: pytest.MonkeyPatch) -> None:
    seq = iter([
        (2000, None),
        (500, "c1"),
        (500, "c1"),
    ])
    monkeypatch.setattr(catch_up, "_eligible_count", lambda: next(seq))
    train = MagicMock(return_value=_ok_result())
    promote = MagicMock()
    monkeypatch.setattr(catch_up, "run_baseline_training_pipeline", train)
    monkeypatch.setattr(catch_up, "run_baseline_promotion_pipeline", promote)

    assert catch_up.loop_until_caught_up(promote=False, max_rounds=5) == 0
    assert train.call_count == 1
    promote.assert_not_called()


def test_loop_train_failure_returns_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(catch_up, "_eligible_count", lambda: (5000, None))
    monkeypatch.setattr(
        catch_up,
        "run_baseline_training_pipeline",
        MagicMock(return_value=_fail_result()),
    )
    promote = MagicMock()
    monkeypatch.setattr(catch_up, "run_baseline_promotion_pipeline", promote)

    assert catch_up.loop_until_caught_up(promote=True, max_rounds=5) == 1
    promote.assert_not_called()


def test_loop_stall_when_count_unchanged_returns_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(catch_up, "_eligible_count", lambda: (5000, "same"))
    monkeypatch.setattr(
        catch_up,
        "run_baseline_training_pipeline",
        MagicMock(return_value=_ok_result()),
    )
    promote = MagicMock()
    monkeypatch.setattr(catch_up, "run_baseline_promotion_pipeline", promote)

    assert catch_up.loop_until_caught_up(promote=True, max_rounds=5) == 2
    promote.assert_not_called()


def test_loop_max_rounds_returns_three(monkeypatch: pytest.MonkeyPatch) -> None:
    n = {"v": 10_000}

    def eligible() -> tuple[int, object]:
        return n["v"], f"c{n['v']}"

    def train() -> PipelineRunResult:
        n["v"] -= 1
        return _ok_result()

    promote = MagicMock(return_value=_ok_result(name="baseline_promotion"))
    monkeypatch.setattr(catch_up, "_eligible_count", eligible)
    monkeypatch.setattr(catch_up, "run_baseline_training_pipeline", train)
    monkeypatch.setattr(catch_up, "run_baseline_promotion_pipeline", promote)

    assert catch_up.loop_until_caught_up(promote=True, max_rounds=3) == 3
    assert promote.call_count == 3


def test_loop_promote_failure_returns_one(monkeypatch: pytest.MonkeyPatch) -> None:
    seq = iter([(3000, None), (1000, "c1")])
    monkeypatch.setattr(catch_up, "_eligible_count", lambda: next(seq))
    monkeypatch.setattr(
        catch_up,
        "run_baseline_training_pipeline",
        MagicMock(return_value=_ok_result()),
    )
    monkeypatch.setattr(
        catch_up,
        "run_baseline_promotion_pipeline",
        MagicMock(return_value=_fail_result(name="baseline_promotion")),
    )

    assert catch_up.loop_until_caught_up(promote=True, max_rounds=5) == 1


def test_max_rounds_clamped_to_at_least_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        catch_up,
        "_eligible_count",
        lambda: (MIN_NEW_MOVES_BASELINE - 1, None),
    )
    assert catch_up.loop_until_caught_up(promote=False, max_rounds=0) == 0
