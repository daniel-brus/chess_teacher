"""The daily runner no longer launches a standalone split-assignment pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from chess_teacher.pipelines.runner import PipelineRunner


def test_run_account_returns_ingestion_and_preprocessing_only() -> None:
    user = MagicMock()
    user.user_id = "user-1"
    account = MagicMock()
    account.account_id = "acct-1"
    account.format_label.return_value = "acct"
    runner = PipelineRunner(user=user, db_client=MagicMock())
    ingestion = MagicMock()
    preprocessing = MagicMock()
    pressure = MagicMock()
    pressure.rss_mb = 1.0
    pressure.format_fields.return_value = ""

    with (
        patch(
            "chess_teacher.pipelines.runner.run_ingestion_pipeline",
            return_value=ingestion,
        ) as run_ingestion,
        patch(
            "chess_teacher.pipelines.runner.run_preprocessing_pipeline",
            return_value=preprocessing,
        ) as run_preprocessing,
        patch(
            "chess_teacher.pipelines.runner.snapshot_host_pressure",
            return_value=pressure,
        ),
    ):
        results = runner._run_account(account)

    assert results == [ingestion, preprocessing]
    run_ingestion.assert_called_once()
    run_preprocessing.assert_called_once()
    assert run_preprocessing.call_args.args[:2] == ("user-1", account)
