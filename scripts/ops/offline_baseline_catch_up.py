"""Thin CLI: offline catch-up sibling (no TrainingState / baseline_models write).

Run (dev)::

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/ops/offline_baseline_catch_up.py

Optional::

    --split-version baseline-v1
    --limit 10000
    --full-val
    --max-rounds 50
    --min-new-moves 1000
    --batch-limit 10000
    --start-cutoff 2026-01-01T00:00:00
    --start-from-production-cutoff
    --parent-uri PATH_OR_URI
    --epochs 20
    --style-disagree-boost 2.0
    --style-disagree-scale 2.0
    --output-dir DIR
"""

from __future__ import annotations

from chess_teacher.pipelines.neural_network.offline_catch_up import main
from chess_teacher.pipelines.neural_network.tf_runtime import ensure_tensorflow_logging
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.process_utils import log_script_runtime_context, run_script_main

ensure_tensorflow_logging()
logger = get_logger()


if __name__ == "__main__":
    log_script_runtime_context(logger, script="offline_baseline_catch_up")
    run_script_main(main)
