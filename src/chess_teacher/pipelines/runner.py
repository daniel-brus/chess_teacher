from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from chess_teacher.pipelines.ingestion.main import run_ingestion_pipeline
from chess_teacher.platform.account import Account
from chess_teacher.platform.user import User
from chess_teacher.platform.users_accounts import get_accounts_for_user
from chess_teacher.utils.db.client import DatabaseClient
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.pipeline_utils.pipeline_helpers import (
    PipelineRunResult,
    ProgressWindow,
)

logger = get_logger()

_DEFAULT_MAX_ACCOUNT_WORKERS = 4


class PipelineRunner:
    """Top-level orchestrator that runs all pipelines for one user."""

    def __init__(
        self,
        user: User,
        db_client: DatabaseClient,
        *,
        max_account_workers: int = _DEFAULT_MAX_ACCOUNT_WORKERS,
        progress_window: ProgressWindow | None = None,
    ) -> None:
        self.user = user
        self.db_client = db_client
        self.max_account_workers = max_account_workers
        self.progress_window = progress_window

    def run(self) -> list[PipelineRunResult]:
        accounts = get_accounts_for_user(self.user, self.db_client)
        if not accounts:
            logger.info("No accounts linked for user=%s; nothing to run.", self.user.user_id)
            return []

        if self.progress_window is not None:
            return self._run_accounts_sequential(accounts)

        if len(accounts) == 1:
            return [self._run_account(accounts[0])]

        return self._run_accounts_parallel(accounts)

    def _run_account(self, account: Account) -> PipelineRunResult:
        logger.info(
            "Starting ingestion for user=%s account=%s (%s).",
            self.user.user_id,
            account.account_id,
            account.format_label(),
        )
        result = run_ingestion_pipeline(
            self.user.user_id,
            account,
            progress_window=self.progress_window,
        )
        logger.info(
            "Finished ingestion for user=%s account=%s with result=%s.",
            self.user.user_id,
            account.account_id,
            result.result.value,
        )
        return result

    def _run_accounts_sequential(self, accounts: list[Account]) -> list[PipelineRunResult]:
        results: list[PipelineRunResult] = []
        for index, account in enumerate(accounts, start=1):
            if len(accounts) > 1 and self.progress_window is not None:
                self.progress_window.next(
                    f"Account {index}/{len(accounts)}: {account.format_label()}",
                )
            results.append(self._run_account(account))
        return results

    def _run_accounts_parallel(self, accounts: list[Account]) -> list[PipelineRunResult]:
        workers = min(self.max_account_workers, len(accounts))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(self._run_account, accounts))


def run_pipeline(
    user: User,
    db_client: DatabaseClient,
    *,
    max_account_workers: int = _DEFAULT_MAX_ACCOUNT_WORKERS,
    progress_window: ProgressWindow | None = None,
) -> list[PipelineRunResult]:
    """Run all pipelines for one user."""
    return PipelineRunner(
        user,
        db_client,
        max_account_workers=max_account_workers,
        progress_window=progress_window,
    ).run()
