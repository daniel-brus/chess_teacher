"""Unit tests for AssignGameSplitsStep."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from chess_teacher.pipelines.neural_network.split_registry import BackfillResult
from chess_teacher.pipelines.neural_network.split_steps import AssignGameSplitsStep
from chess_teacher.pipelines.neural_network.splits import DEFAULT_SPLIT_SALT
from chess_teacher.utils.pipeline_utils.pipeline_base import PipelineContext


def test_assign_game_splits_requires_account_id() -> None:
    step = AssignGameSplitsStep()
    with pytest.raises(ValueError, match="account_id"):
        step.run(MagicMock(), PipelineContext(user_id="u1", account_id=None))


def test_assign_game_splits_calls_registry_for_account() -> None:
    db = MagicMock()
    context = PipelineContext(user_id="u1", account_id="acct-9")
    result = BackfillResult(
        split_version=DEFAULT_SPLIT_SALT,
        eligible_games=4,
        newly_assigned=1,
        already_assigned=3,
    )
    with patch(
        "chess_teacher.pipelines.neural_network.split_steps.SplitRegistry.ensure_eligible_games_for_account",
        return_value=result,
    ) as ensure:
        AssignGameSplitsStep().run(db, context)
    ensure.assert_called_once_with("acct-9")
    assert context.extras["split_assign"] is result
