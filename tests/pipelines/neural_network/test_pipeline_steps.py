"""Unit tests for baseline train pipeline steps (mocked store/registry)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from chess_teacher.pipelines.neural_network.models import TrainingState
from chess_teacher.pipelines.neural_network.pipeline_steps import (
    MIN_NEW_MOVES_BASELINE,
    CheckSufficientNewDataStep,
    LoadNewDataStep,
    UpdateTrainingStateStep,
)
from chess_teacher.pipelines.neural_network.splits import DEFAULT_SPLIT_SALT
from chess_teacher.utils.pipeline_utils.pipeline_base import PipelineContext


def _context(**extras: object) -> PipelineContext:
    ctx = PipelineContext()
    ctx.extras.update(extras)
    return ctx


def test_check_uses_unprocessed_count_not_count_since() -> None:
    store = MagicMock()
    store.count_unprocessed_train.return_value = MIN_NEW_MOVES_BASELINE - 1
    state = TrainingState(scope="baseline")
    db = MagicMock()
    ctx = _context()

    with (
        patch(
            "chess_teacher.pipelines.neural_network.pipeline_steps.TrainingState.for_baseline",
            return_value=state,
        ),
        patch.object(TrainingState, "save_to_db"),
        patch(
            "chess_teacher.pipelines.neural_network.pipeline_steps.TrainingDataStore",
            return_value=store,
        ),
    ):
        CheckSufficientNewDataStep().run(db, ctx)

    store.count_unprocessed_train.assert_called_once_with(split_version=DEFAULT_SPLIT_SALT)
    store.count_since.assert_not_called()
    assert ctx.extras["baseline_skip"] is True
    assert ctx.extras["new_move_count"] == MIN_NEW_MOVES_BASELINE - 1


def test_check_proceeds_when_queue_meets_floor() -> None:
    store = MagicMock()
    store.count_unprocessed_train.return_value = MIN_NEW_MOVES_BASELINE
    state = TrainingState(scope="baseline")
    db = MagicMock()
    ctx = _context()

    with (
        patch(
            "chess_teacher.pipelines.neural_network.pipeline_steps.TrainingState.for_baseline",
            return_value=state,
        ),
        patch.object(TrainingState, "save_to_db"),
        patch(
            "chess_teacher.pipelines.neural_network.pipeline_steps.TrainingDataStore",
            return_value=store,
        ),
    ):
        CheckSufficientNewDataStep().run(db, ctx)

    assert ctx.extras["baseline_skip"] is False


def test_load_uses_queue_fetch_not_fetch_since() -> None:
    store = MagicMock()
    datums = [MagicMock(game_id="g1")]
    store.fetch_unprocessed_train_batch.return_value = (datums, ["g1"])
    db = MagicMock()
    ctx = _context(baseline_skip=False, split_version=DEFAULT_SPLIT_SALT)

    with patch(
        "chess_teacher.pipelines.neural_network.pipeline_steps.TrainingDataStore",
        return_value=store,
    ):
        LoadNewDataStep().run(db, ctx)

    store.fetch_unprocessed_train_batch.assert_called_once()
    kwargs = store.fetch_unprocessed_train_batch.call_args.kwargs
    assert kwargs["split_version"] == DEFAULT_SPLIT_SALT
    store.fetch_since.assert_not_called()
    assert ctx.extras["training_datums"] == datums
    assert ctx.extras["batch_game_ids"] == ["g1"]


def test_load_skip_does_not_fetch() -> None:
    store = MagicMock()
    db = MagicMock()
    ctx = _context(baseline_skip=True)

    with patch(
        "chess_teacher.pipelines.neural_network.pipeline_steps.TrainingDataStore",
        return_value=store,
    ):
        LoadNewDataStep().run(db, ctx)

    store.fetch_unprocessed_train_batch.assert_not_called()
    store.fetch_since.assert_not_called()


def test_update_marks_game_ids_after_success() -> None:
    registry = MagicMock()
    registry.mark_processed.return_value = 2
    db = MagicMock()
    ctx = _context(
        baseline_skip=False,
        batch_game_ids=["g1", "g2"],
        split_version=DEFAULT_SPLIT_SALT,
    )

    with patch(
        "chess_teacher.pipelines.neural_network.pipeline_steps.SplitRegistry",
        return_value=registry,
    ) as registry_cls:
        UpdateTrainingStateStep().run(db, ctx)

    registry_cls.assert_called_once_with(db, split_version=DEFAULT_SPLIT_SALT)
    registry.mark_processed.assert_called_once_with(["g1", "g2"])
    assert ctx.extras["marked_game_count"] == 2


def test_update_skip_does_not_mark() -> None:
    registry = MagicMock()
    db = MagicMock()
    ctx = _context(baseline_skip=True, batch_game_ids=["g1"])

    with patch(
        "chess_teacher.pipelines.neural_network.pipeline_steps.SplitRegistry",
        return_value=registry,
    ):
        UpdateTrainingStateStep().run(db, ctx)

    registry.mark_processed.assert_not_called()
