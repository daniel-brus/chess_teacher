"""Preprocessing ends by assigning this account's games to the split registry."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from chess_teacher.pipelines.neural_network.split_steps import AssignGameSplitsStep
from chess_teacher.pipelines.preprocessing.main import run_preprocessing_pipeline
from chess_teacher.pipelines.preprocessing.pipeline_steps import (
    EnrichExpensiveMoveCharacteristicsStep,
)


def test_preprocessing_pipeline_assigns_splits_last() -> None:
    account = MagicMock()
    account.account_id = "acct-1"
    with patch("chess_teacher.pipelines.preprocessing.main.Pipeline") as pipeline_cls:
        pipeline_cls.return_value.run.return_value = MagicMock()
        run_preprocessing_pipeline("user-1", account)

    kwargs = pipeline_cls.call_args.kwargs
    assert kwargs["name"] == "preprocessing"
    assert kwargs["user_id"] == "user-1"
    assert kwargs["account_id"] == "acct-1"
    steps = kwargs["steps"]
    assert isinstance(steps[-2], EnrichExpensiveMoveCharacteristicsStep)
    assert isinstance(steps[-1], AssignGameSplitsStep)
