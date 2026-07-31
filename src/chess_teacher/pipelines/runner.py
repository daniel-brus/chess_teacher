from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from chess_teacher.pipelines.ingestion.main import run_ingestion_pipeline
from chess_teacher.pipelines.modes import PipelineMode
from chess_teacher.pipelines.preprocessing.main import run_preprocessing_pipeline
from chess_teacher.platform.account import Account
from chess_teacher.platform.user import User
from chess_teacher.utils.db.client import DatabaseClient
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.pipeline_utils.pipeline_helpers import (
    PipelineRunResult,
    ProgressWindow,
)

logger = get_logger()

_DEFAULT_MAX_ACCOUNT_WORKERS = 4


class PipelineRunner:
    """Top-level orchestrator that runs ingestion then preprocessing per account."""

    def __init__(
        self,
        user: User,
        db_client: DatabaseClient,
        *,
        max_account_workers: int = _DEFAULT_MAX_ACCOUNT_WORKERS,
        mode: PipelineMode = PipelineMode.INCREMENTAL,
        progress_window: ProgressWindow | None = None,
    ) -> None:
        self.user = user
        self.db_client = db_client
        self.max_account_workers = max_account_workers
        self.mode = mode
        self.progress_window = progress_window

    def run(self) -> list[PipelineRunResult]:
        accounts = self.user.get_linked_accounts(self.db_client)
        if not accounts:
            logger.info("No accounts linked for user=%s; nothing to run.", self.user.user_id)
            return []

        if self.progress_window is not None:
            return self._run_accounts_sequential(accounts)

        if len(accounts) == 1:
            return self._run_account(accounts[0])

        return self._run_accounts_parallel(accounts)

    def _run_account(self, account: Account) -> list[PipelineRunResult]:
        logger.info(
            "Starting ingestion for user=%s account=%s (%s).",
            self.user.user_id,
            account.account_id,
            account.format_label(),
        )
        ingestion_result = run_ingestion_pipeline(
            self.user.user_id,
            account,
            mode=self.mode,
            progress_window=self.progress_window,
        )
        logger.info(
            "Finished ingestion for user=%s account=%s with result=%s.",
            self.user.user_id,
            account.account_id,
            ingestion_result.result.value,
        )

        logger.info(
            "Starting preprocessing for user=%s account=%s (%s).",
            self.user.user_id,
            account.account_id,
            account.format_label(),
        )
        preprocessing_result = run_preprocessing_pipeline(
            self.user.user_id,
            account,
            mode=self.mode,
            progress_window=self.progress_window,
        )
        logger.info(
            "Finished preprocessing for user=%s account=%s with result=%s.",
            self.user.user_id,
            account.account_id,
            preprocessing_result.result.value,
        )
        # Follow-up: run_user_finetune_pipeline(self.user.user_id) after baseline exists.
        return [ingestion_result, preprocessing_result]

    def _run_accounts_sequential(self, accounts: list[Account]) -> list[PipelineRunResult]:
        results: list[PipelineRunResult] = []
        for index, account in enumerate(accounts, start=1):
            if len(accounts) > 1 and self.progress_window is not None:
                self.progress_window.next(
                    f"Account {index}/{len(accounts)}: {account.format_label()}",
                )
            results.extend(self._run_account(account))
        return results

    def _run_accounts_parallel(self, accounts: list[Account]) -> list[PipelineRunResult]:
        workers = min(self.max_account_workers, len(accounts))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            nested = list(executor.map(self._run_account, accounts))
        return [result for account_results in nested for result in account_results]


def run_pipeline(
    user: User,
    db_client: DatabaseClient,
    *,
    max_account_workers: int = _DEFAULT_MAX_ACCOUNT_WORKERS,
    mode: PipelineMode = PipelineMode.INCREMENTAL,
    progress_window: ProgressWindow | None = None,
) -> list[PipelineRunResult]:
    """Run ingestion then preprocessing for every linked account."""
    return PipelineRunner(
        user,
        db_client,
        max_account_workers=max_account_workers,
        mode=mode,
        progress_window=progress_window,
    ).run()
