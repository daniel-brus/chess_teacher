"""User training runs once per dispatched pipeline, not once per account."""

from __future__ import annotations

from unittest.mock import MagicMock

from chess_teacher.pipelines.runner import PipelineRunner


def test_user_training_runs_once_after_all_accounts(monkeypatch) -> None:
    user = MagicMock()
    user.user_id = "user-1"
    user.get_linked_accounts.return_value = [
        MagicMock(account_id="acct-a"),
        MagicMock(account_id="acct-b"),
    ]
    runner = PipelineRunner(user, MagicMock())
    monkeypatch.setattr(runner, "_run_account", lambda account: [MagicMock()])
    train = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(
        "chess_teacher.pipelines.runner.run_personal_training_pipeline",
        train,
    )

    results = runner.run()

    train.assert_called_once_with("user-1", promote=True, progress_window=None)
    assert len(results) == 3


def test_user_training_skips_when_there_are_no_accounts(monkeypatch) -> None:
    user = MagicMock()
    user.user_id = "user-1"
    user.get_linked_accounts.return_value = []
    runner = PipelineRunner(user, MagicMock())
    train = MagicMock()
    monkeypatch.setattr(
        "chess_teacher.pipelines.runner.run_personal_training_pipeline",
        train,
    )

    assert runner.run() == []
    train.assert_not_called()
