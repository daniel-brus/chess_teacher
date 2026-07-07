from __future__ import annotations

import argparse
import sys

from chess_teacher.pipelines.modes import PIPELINE_MODES, PipelineMode
from chess_teacher.pipelines.runner import run_pipeline
from chess_teacher.platform.user import User
from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.pipeline_utils.json_lines_progress import JsonLinesProgressWindow
from chess_teacher.utils.pipeline_utils.pipeline_helpers import aggregate_pipeline_run_results

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
            "+ full_sync deletes in preprocessing and raw_games load)."
        ),
    )
    parser.add_argument(
        "--progress-stdout",
        action="store_true",
        help="Emit ProgressWindow events as JSON lines on stdout (for UI subprocess hosts).",
    )
    args = parser.parse_args()

    db_client = get_db_client()
    user = User.fetch_from_db(db_client, id=args.user_id)

    progress_window = JsonLinesProgressWindow(sys.stdout) if args.progress_stdout else None

    logger.info("Pipeline job started for user=%s (mode=%s).", user.user_id, args.mode)
    results = run_pipeline(user, db_client, mode=args.mode, progress_window=progress_window)

    aggregated = aggregate_pipeline_run_results(results)
    if aggregated.latest_successful_run_id is not None:
        user.update_latest_pipeline_run(db_client, aggregated.latest_successful_run_id)

    account_count = len({result.account_id for result in results if result.account_id is not None})
    logger.info(
        "Pipeline job finished for user=%s: accounts=%s pipeline_runs=%s results=%s.",
        user.user_id,
        account_count,
        len(results),
        [result.result.value for result in results],
    )
    return 0


if __name__ == "__main__":
    from chess_teacher.utils.process_utils import run_script_main

    run_script_main(main)
