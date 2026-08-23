"""Reset baseline training state for a cold restart.

Clears ``training_state.last_trained_data_cutoff`` and archives all
non-archived ``baseline_models`` rows. MLflow artifacts are kept.

Run (dev)::

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/baseline_reset_training.py --dry-run

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/baseline_reset_training.py --yes

Prod (after backfill, from a host that can reach prod Postgres)::

    doppler run --project chess-teacher --config prod -- ^
      .venv\\Scripts\\python.exe scripts/baseline_reset_training.py --yes
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
    if result.cutoff_cleared:
        print(f"{prefix} clear cutoff (was {result.previous_cutoff})")
    if result.models_archived:
        versions = ", ".join(result.archived_versions)
        print(f"{prefix} archive {result.models_archived} baseline(s): {versions}")
    if not result.cutoff_cleared and not result.models_archived:
        print("Nothing to reset (cutoff already NULL and no active baselines).")


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
        "--no-clear-cutoff",
        action="store_true",
        help="Only archive baseline model rows.",
    )
    parser.add_argument(
        "--no-archive-models",
        action="store_true",
        help="Only clear the training cutoff.",
    )
    args = parser.parse_args()
    log_script_runtime_context(logger, script="baseline_reset_training")

    db = get_db_client()
    preview = reset_baseline_training(
        db,
        clear_cutoff=not args.no_clear_cutoff,
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
        clear_cutoff=not args.no_clear_cutoff,
        archive_models=not args.no_archive_models,
        dry_run=False,
    )
    _print_result(dry_run=False, result=result)
    return 0


if __name__ == "__main__":
    run_script_main(main)
