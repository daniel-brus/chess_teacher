"""Reset platform and personal training queue markers.

Clears both processed markers and retry counters on train-bucket rows for the
current split. Model rows, training cutoffs, and validation/test assignments
are preserved.

Run (dev)::

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/ops/training_queue_reset.py --dry-run

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/ops/training_queue_reset.py --yes

Production (on the VPS, after this reset implementation is deployed)::

    /opt/chess_teacher/k8s/run-script-job.sh training_queue_reset -- --dry-run --wait
    /opt/chess_teacher/k8s/run-script-job.sh training_queue_reset -- --yes --wait
"""

from __future__ import annotations

import argparse
import sys

from chess_teacher.pipelines.neural_network.training_queue_reset import reset_training_queues
from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.process_utils import log_script_runtime_context, run_script_main

logger = get_logger()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report affected train rows without changing queue markers.",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Apply without interactive confirmation.",
    )
    args = parser.parse_args(argv)
    log_script_runtime_context(logger, script="training_queue_reset")

    db = get_db_client()
    count = reset_training_queues(db, dry_run=True)
    print(f"Would reset both training queues on {count} train rows.")

    if args.dry_run:
        return 0

    if not args.yes:
        if not sys.stdin.isatty():
            logger.error("Refusing to reset without --yes (stdin is not a TTY).")
            return 1
        answer = input(f"Reset both training queues on {count} train rows? [y/N] ")
        answer = answer.strip().lower()
        if answer not in {"y", "yes"}:
            print("Aborted.")
            return 1

    updated = reset_training_queues(db)
    print(f"Reset both training queues on {updated} train rows.")
    return 0


if __name__ == "__main__":
    run_script_main(main)
