"""Reset baseline training state for a cold restart.

Archives all non-archived ``baseline_models`` rows and clears
``already_processed_baseline``. MLflow artifacts are kept.

Run (dev)::

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/ops/baseline_reset_training.py --dry-run

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/ops/baseline_reset_training.py --yes

Prod (after backfill, from a host that can reach prod Postgres)::

    doppler run --project chess-teacher --config prod -- ^
      .venv\\Scripts\\python.exe scripts/ops/baseline_reset_training.py --yes
"""

from __future__ import annotations

import argparse
import sys

from chess_teacher.pipelines.neural_network.baseline_reset import reset_baseline_training
from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.process_utils import log_script_runtime_context, run_script_main

logger = get_logger()


def _print_result(*, dry_run: bool, result) -> None:
    prefix = "Would" if dry_run else "Did"
    if result.models_archived:
        versions = ", ".join(result.archived_versions)
        print(f"{prefix} archive {result.models_archived} baseline(s): {versions}")
    if not dry_run:
        print(f"{prefix} clear baseline processed flags ({result.flags_cleared} rows)")
    elif not result.models_archived:
        print("Would clear baseline processed flags (row count after apply).")
    if dry_run and result.models_archived:
        print("Would also clear baseline processed flags.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing to the database.",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Apply without interactive confirmation.",
    )
    parser.add_argument(
        "--no-archive-models",
        action="store_true",
        help="Only clear baseline processed flags.",
    )
    args = parser.parse_args()
    log_script_runtime_context(logger, script="baseline_reset_training")

    db = get_db_client()
    preview = reset_baseline_training(
        db,
        archive_models=not args.no_archive_models,
        dry_run=True,
    )
    _print_result(dry_run=True, result=preview)

    if args.dry_run:
        return 0

    if not args.yes:
        if not sys.stdin.isatty():
            logger.error("Refusing to reset without --yes (stdin is not a TTY).")
            return 1
        answer = input("Apply baseline reset? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Aborted.")
            return 1

    result = reset_baseline_training(
        db,
        archive_models=not args.no_archive_models,
        dry_run=False,
    )
    _print_result(dry_run=False, result=result)
    return 0


if __name__ == "__main__":
    run_script_main(main)
