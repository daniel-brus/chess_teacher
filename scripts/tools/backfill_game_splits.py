"""Backfill persistent game split assignments (Phase 1b).

Assigns every training-eligible ``game_id`` to train/val/test for a ``split_version``
(salt) and stores rows in ``ml.game_split_assignments``. Idempotent.

Daily catch-up: ``AssignGameSplitsStep`` in the per-account user pipeline
(``PipelineRunner`` after preprocess). Use this script for empty DBs or
platform-wide repair.

Run (dev)::

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/tools/backfill_game_splits.py

Optional::

    --split-version baseline-v1
    --batch-size 500
"""

from __future__ import annotations

import argparse

from chess_teacher.pipelines.neural_network.split_registry import get_split_registry
from chess_teacher.pipelines.neural_network.splits import DEFAULT_SPLIT_SALT
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.process_utils import log_script_runtime_context, run_script_main

logger = get_logger()


def run_backfill(*, split_version: str, batch_size: int) -> int:
    registry = get_split_registry(split_version=split_version)
    logger.info(
        "Starting split backfill split_version=%s batch_size=%s…",
        split_version,
        batch_size,
    )
    result = registry.backfill_eligible_games(batch_size=batch_size)
    print("\n=== split backfill summary ===")
    print(f"split_version={result.split_version!r}")
    print(f"eligible_games={result.eligible_games}")
    print(f"newly_assigned={result.newly_assigned}")
    print(f"already_assigned={result.already_assigned}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-version", type=str, default=DEFAULT_SPLIT_SALT)
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    log_script_runtime_context(logger, script="backfill_game_splits")
    return run_backfill(
        split_version=str(args.split_version),
        batch_size=max(1, int(args.batch_size)),
    )


if __name__ == "__main__":
    run_script_main(main)
