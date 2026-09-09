"""CLI: train (+ promote) until baseline eligible backlog is below the floor.

Stops when ``count_since(cutoff) < MIN_NEW_MOVES_BASELINE`` (default 1000).

Run (dev)::

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/ops/baseline_train_until_caught_up.py

Optional::

    --no-promote
    --max-rounds N
"""

from __future__ import annotations

import argparse

from chess_teacher.pipelines.neural_network.catch_up import loop_until_caught_up
from chess_teacher.pipelines.neural_network.tf_runtime import ensure_tensorflow_logging
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.process_utils import log_script_runtime_context, run_script_main

ensure_tensorflow_logging()
logger = get_logger()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Skip promotion after each successful train round.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=50,
        help="Safety cap on train(+promote) iterations (default 50).",
    )
    args = parser.parse_args()
    log_script_runtime_context(logger, script="baseline_train_until_caught_up")
    return loop_until_caught_up(
        promote=not args.no_promote,
        max_rounds=max(1, args.max_rounds),
    )


if __name__ == "__main__":
    run_script_main(main)
