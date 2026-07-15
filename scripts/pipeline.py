from __future__ import annotations

import argparse
import sys
import time

from chess_teacher.pipelines.modes import PIPELINE_MODES, PipelineMode
from chess_teacher.pipelines.runner import run_pipeline
from chess_teacher.platform.user import User
from chess_teacher.utils.cache_utils import invalidate_user_games_and_accounts_cache
from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.pipeline_utils.json_lines_progress import JsonLinesProgressWindow
from chess_teacher.utils.pipeline_utils.pipeline_helpers import (
    PipelineResult,
    aggregate_pipeline_run_results,
)
from chess_teacher.utils.process_utils import log_script_runtime_context

logger = get_logger()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pipelines for one user.")
    parser.add_argument("--user-id", required=True)
    parser.add_argument(
        "--mode",
        type=PipelineMode,
        choices=PIPELINE_MODES,
        default=PipelineMode.INCREMENTAL,
        help=(
            "Pipeline run mode: incremental (default), retry (re-load ingested+failed), "
            "reprocess (re-load all storage folders + upsert), or full_reload (same folders "
            "+ full_sync deletes in preprocessing and raw_games load."
        ),
    )
    parser.add_argument(
        "--progress-stdout",
        action="store_true",
        help="Emit ProgressWindow events as JSON lines on stdout (for UI subprocess hosts).",
    )
    args = parser.parse_args()

    log_script_runtime_context(logger, script="pipeline")
    logger.info(
        "Pipeline job starting user_id=%s mode=%s progress_stdout=%s",
        args.user_id,
        args.mode,
        args.progress_stdout,
    )

    db_client = get_db_client()
    user = User.fetch_from_db(db_client, id=args.user_id)
    accounts = user.get_linked_accounts(db_client)
    logger.info(
        "Pipeline job linked accounts: count=%s account_ids=%s",
        len(accounts),
        [account.account_id for account in accounts],
    )

    progress_window = JsonLinesProgressWindow(sys.stdout) if args.progress_stdout else None

    started_at = time.monotonic()
    results = run_pipeline(user, db_client, mode=args.mode, progress_window=progress_window)
    duration_s = time.monotonic() - started_at

    aggregated = aggregate_pipeline_run_results(results)
    if aggregated.latest_successful_run_id is not None:
        user.update_latest_pipeline_run(db_client, aggregated.latest_successful_run_id)
        logger.info(
            "Pipeline job updated latest_pipeline_run user_id=%s run_id=%s",
            user.user_id,
            aggregated.latest_successful_run_id,
        )

    invalidate_user_games_and_accounts_cache(user.user_id)

    account_count = len({result.account_id for result in results if result.account_id is not None})
    logger.info(
        "Pipeline job finished user_id=%s aggregate_result=%s duration_s=%.1f accounts=%s "
        "pipeline_runs=%s run_ids=%s step_results=%s",
        user.user_id,
        aggregated.result.value,
        duration_s,
        account_count,
        len(results),
        list(aggregated.run_ids),
        [result.result.value for result in results],
    )
    for run_result in results:
        logger.info(
            "Pipeline run summary name=%s account_id=%s result=%s duration_s=%.1f run_id=%s",
            run_result.name,
            run_result.account_id,
            run_result.result.value,
            run_result.duration_seconds,
            run_result.run_id,
        )
    if aggregated.result in {PipelineResult.FAILURE, PipelineResult.PARTIAL}:
        for error_message in aggregated.error_messages:
            logger.warning("Pipeline step error: %s", error_message)

    return 0


if __name__ == "__main__":
    from chess_teacher.utils.process_utils import run_script_main

    run_script_main(main)
