"""Start a new training tree after switching model architecture.

Default: archives baseline and personal rows that are **not** the hybrid
board+state scheme, clears the parent-baseline pointer on those rows, resets
both training queues, and clears personal_queue_baseline. Hybrid rows stay.

Nuclear (``--all``): archives **every** non-archived baseline and personal
row (including hybrid), clears any parent-baseline pointer, then the same
queue / personal_queue_baseline resets, and stamps ``model_tree_started_at``
so the Training page hides pre-cutoff lineage. Use when the served graph is
rotten even though metrics claim hybrid.

``--stamp-lineage-cutoff``: only stamps the Training cutoff (earliest live
baseline ``trained_at``, or now). No archive. Use after a nuclear that ran
before this field existed.

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

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/ops/start_model_tree.py --stamp-lineage-cutoff --yes

Production (on the VPS, after this script is deployed)::

    /opt/chess_teacher/k8s/run-script-job.sh start_model_tree -- --all --dry-run --wait
    /opt/chess_teacher/k8s/run-script-job.sh start_model_tree -- --all --yes --wait
    /opt/chess_teacher/k8s/run-script-job.sh start_model_tree -- --stamp-lineage-cutoff --yes --wait
"""

from __future__ import annotations

import argparse
import sys

from chess_teacher.bots.presets import invalidate_baseline_presets_cache
from chess_teacher.pipelines.neural_network.model_tree import (
    format_model_tree_reset,
    stamp_model_tree_started_at,
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
            "personal model (including hybrid), clear any parent baseline, "
            "and stamp the Training lineage cutoff."
        ),
    )
    parser.add_argument(
        "--stamp-lineage-cutoff",
        action="store_true",
        help=(
            "Only stamp TrainingState.model_tree_started_at to the earliest "
            "live baseline trained_at (or now). No archive."
        ),
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Apply without interactive confirmation.",
    )
    args = parser.parse_args(argv)
    if args.all and args.stamp_lineage_cutoff:
        parser.error("Use either --all or --stamp-lineage-cutoff, not both.")
    log_script_runtime_context(logger, script="start_model_tree")

    db = get_db_client()

    if args.stamp_lineage_cutoff:
        preview_at = stamp_model_tree_started_at(db, dry_run=True)
        print(
            f"Would stamp model_tree_started_at={preview_at.isoformat()} "
            "(Training page hides older lineage)."
        )
        if args.dry_run:
            return 0
        if not args.yes:
            if not sys.stdin.isatty():
                logger.error("Refusing to stamp lineage cutoff without --yes (stdin is not a TTY).")
                return 1
            answer = input("Stamp Training lineage cutoff? [y/N] ")
            if answer.strip().lower() not in {"y", "yes"}:
                print("Aborted.")
                return 1
        stamped = stamp_model_tree_started_at(db)
        print(
            f"Stamped model_tree_started_at={stamped.isoformat()} "
            "(Training page hides older lineage)."
        )
        return 0

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
