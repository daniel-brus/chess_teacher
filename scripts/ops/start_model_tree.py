"""Start a new training tree after switching model architecture.

Default: archives baseline and personal rows that are **not** the hybrid
board+state scheme, clears the parent-baseline pointer on those rows, resets
both training queues, and clears personal_queue_baseline. Hybrid rows stay.

Nuclear (``--all``): archives **every** non-archived baseline and personal
row (including hybrid), clears any parent-baseline pointer, then the same
queue / personal_queue_baseline resets. Use when the served graph is rotten
even though metrics claim hybrid.

Archived models stay playable. MLflow artifacts are not deleted.

Run this once when the served architecture changes, or with ``--all`` for a
full soft reset. It is intentionally not part of the nightly trainer.

Run (dev)::

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/ops/start_model_tree.py --dry-run

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/ops/start_model_tree.py --all --dry-run

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/ops/start_model_tree.py --all --yes

Production (on the VPS, after this script is deployed)::

    /opt/chess_teacher/k8s/run-script-job.sh start_model_tree -- --all --dry-run --wait
    /opt/chess_teacher/k8s/run-script-job.sh start_model_tree -- --all --yes --wait
"""

from __future__ import annotations

import argparse
import sys

from chess_teacher.bots.presets import invalidate_baseline_presets_cache
from chess_teacher.pipelines.neural_network.model_tree import (
    format_model_tree_reset,
    start_new_model_tree,
)
from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.process_utils import log_script_runtime_context, run_script_main

logger = get_logger()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report models, queue rows, and scopes without writing.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Nuclear soft reset: archive every non-archived baseline and "
            "personal model (including hybrid) and clear any parent baseline."
        ),
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Apply without interactive confirmation.",
    )
    args = parser.parse_args(argv)
    log_script_runtime_context(logger, script="start_model_tree")

    db = get_db_client()
    preview = start_new_model_tree(db, dry_run=True, archive_all=args.all)
    print(format_model_tree_reset(preview, applied=False))

    if args.dry_run:
        return 0

    if not args.yes:
        if not sys.stdin.isatty():
            logger.error("Refusing to start a new model tree without --yes (stdin is not a TTY).")
            return 1
        prompt = (
            "Archive ALL models (nuclear) and reset both training queues? [y/N] "
            if args.all
            else "Archive non-hybrid models and reset both training queues? [y/N] "
        )
        answer = input(prompt)
        if answer.strip().lower() not in {"y", "yes"}:
            print("Aborted.")
            return 1

    applied = start_new_model_tree(db, archive_all=args.all)
    invalidate_baseline_presets_cache()
    print(format_model_tree_reset(applied, applied=True))
    return 0


if __name__ == "__main__":
    run_script_main(main)
