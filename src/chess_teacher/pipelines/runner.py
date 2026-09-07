from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from chess_teacher.pipelines.ingestion.main import run_ingestion_pipeline
from chess_teacher.pipelines.modes import PipelineMode
from chess_teacher.pipelines.neural_network.main import run_assign_game_splits_pipeline
from chess_teacher.pipelines.preprocessing.main import run_preprocessing_pipeline
from chess_teacher.platform.account import Account
from chess_teacher.platform.user import User
from chess_teacher.utils.db.client import DatabaseClient
from chess_teacher.utils.env_utils import get_optional_env_variable
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.pipeline_utils.pipeline_helpers import (
    PipelineRunResult,
    ProgressWindow,
)
from chess_teacher.utils.process_utils import snapshot_host_pressure

logger = get_logger()

# Cap account-level ThreadPool concurrency (small VPS / OOM-prone hosts).
_DEFAULT_MAX_ACCOUNT_WORKERS = 2
_ENV_MAX_ACCOUNT_WORKERS = "MAX_ACCOUNT_WORKERS"


def resolve_max_account_workers(
    *,
    explicit: int | None = None,
    env_value: str | None = None,
) -> int:
    """Resolve account-worker count: explicit arg > ``MAX_ACCOUNT_WORKERS`` env > default 2."""
    if explicit is not None:
        return max(1, int(explicit))
    raw = (
        env_value if env_value is not None else get_optional_env_variable(_ENV_MAX_ACCOUNT_WORKERS)
    )
    if raw.strip():
        return max(1, int(raw.strip()))
    return _DEFAULT_MAX_ACCOUNT_WORKERS


class PipelineRunner:
    """Top-level orchestrator: ingestion → preprocessing → game-split assignment per account."""

    def __init__(
        self,
        user: User,
        db_client: DatabaseClient,
        *,
        max_account_workers: int | None = None,
        mode: PipelineMode = PipelineMode.INCREMENTAL,
        progress_window: ProgressWindow | None = None,
    ) -> None:
        self.user = user
        self.db_client = db_client
        self.max_account_workers = resolve_max_account_workers(explicit=max_account_workers)
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
        account_started = snapshot_host_pressure()
        account_t0 = time.monotonic()
        logger.info(
            "Starting ingestion for user=%s account=%s (%s). %s",
            self.user.user_id,
            account.account_id,
            account.format_label(),
            account_started.format_fields(),
        )
        ingestion_t0 = time.monotonic()
        ingestion_result = run_ingestion_pipeline(
            self.user.user_id,
            account,
            mode=self.mode,
            progress_window=self.progress_window,
        )
        logger.info(
            "Finished ingestion for user=%s account=%s with result=%s duration_s=%.2f.",
            self.user.user_id,
            account.account_id,
            ingestion_result.result.value,
            time.monotonic() - ingestion_t0,
        )

        logger.info(
            "Starting preprocessing for user=%s account=%s (%s).",
            self.user.user_id,
            account.account_id,
            account.format_label(),
        )
        preprocessing_t0 = time.monotonic()
        preprocessing_result = run_preprocessing_pipeline(
            self.user.user_id,
            account,
            mode=self.mode,
            progress_window=self.progress_window,
        )
        logger.info(
            "Finished preprocessing for user=%s account=%s with result=%s duration_s=%.2f.",
            self.user.user_id,
            account.account_id,
            preprocessing_result.result.value,
            time.monotonic() - preprocessing_t0,
        )

        logger.info(
            "Starting game-split assignment for user=%s account=%s (%s).",
            self.user.user_id,
            account.account_id,
            account.format_label(),
        )
        split_t0 = time.monotonic()
        split_result = run_assign_game_splits_pipeline(
            self.user.user_id,
            account,
            progress_window=self.progress_window,
        )
        ended = snapshot_host_pressure()
        logger.info(
            "Finished game-split assignment for user=%s account=%s with result=%s duration_s=%.2f.",
            self.user.user_id,
            account.account_id,
            split_result.result.value,
            time.monotonic() - split_t0,
        )
        logger.info(
            "Finished account pipeline user=%s account=%s duration_s=%.2f delta_rss_mb=%.1f %s",
            self.user.user_id,
            account.account_id,
            time.monotonic() - account_t0,
            ended.rss_mb - account_started.rss_mb,
            ended.format_fields(),
        )
        # Follow-up: run_user_finetune_pipeline(self.user.user_id) after baseline exists.
        return [ingestion_result, preprocessing_result, split_result]

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
        logger.info(
            "Running %s account(s) with max_account_workers=%s (pool=%s). %s",
            len(accounts),
            self.max_account_workers,
            workers,
            snapshot_host_pressure().format_fields(),
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            nested = list(executor.map(self._run_account, accounts))
        return [result for account_results in nested for result in account_results]


def run_pipeline(
    user: User,
    db_client: DatabaseClient,
    *,
    max_account_workers: int | None = None,
    mode: PipelineMode = PipelineMode.INCREMENTAL,
    progress_window: ProgressWindow | None = None,
) -> list[PipelineRunResult]:
    """Run ingestion, preprocessing, then game-split assignment for every linked account."""
    return PipelineRunner(
        user,
        db_client,
        max_account_workers=max_account_workers,
        mode=mode,
        progress_window=progress_window,
    ).run()
