"""Tests for the training queue reset operation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from chess_teacher.pipelines.neural_network.split_registry import SplitRegistry
from chess_teacher.pipelines.neural_network.training_queue_reset import reset_training_queues
from scripts.ops.training_queue_reset import main


def test_reset_resets_both_queue_markers_for_train_rows_only() -> None:
    db = MagicMock()
    db.get_row_count.return_value = 12
    db.update_where.return_value = 12

    result = reset_training_queues(db)

    assert result == 12
    where = db.get_row_count.call_args.kwargs["where"]
    assert "\"bucket\" = 'train'" in where
    assert '"already_processed_baseline" IS NOT NULL' in where
    assert '"already_processed_personal" IS NOT NULL' in where
    assert '"baseline_train_attempts" <> 0' in where
    assert '"personal_train_attempts" <> 0' in where
    assert db.update_where.call_args.args[1] == {
        "already_processed_baseline": None,
        "already_processed_personal": None,
        "baseline_train_attempts": 0,
        "personal_train_attempts": 0,
    }
    db.ensure_metadata.assert_called_once()


def test_reset_dry_run_counts_rows_without_writing() -> None:
    db = MagicMock()
    db.get_row_count.return_value = 7

    result = reset_training_queues(db, dry_run=True)

    assert result == 7
    db.update_where.assert_not_called()


def test_reset_cli_dry_run_reports_row_count(monkeypatch, capsys) -> None:
    db = object()
    monkeypatch.setattr(
        "scripts.ops.training_queue_reset.get_db_client",
        lambda: db,
    )
    with patch(
        "scripts.ops.training_queue_reset.reset_training_queues",
        return_value=5,
    ) as reset:
        assert main(["--dry-run"]) == 0

    reset.assert_called_once_with(db, dry_run=True)
    assert "5 train rows" in capsys.readouterr().out


def test_registry_dry_run_never_updates() -> None:
    db = MagicMock()
    db.get_row_count.return_value = 0

    assert SplitRegistry(db).reset_training_queue_markers(dry_run=True) == 0

    db.update_where.assert_not_called()
