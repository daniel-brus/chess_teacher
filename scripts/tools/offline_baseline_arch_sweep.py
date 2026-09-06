"""Thin CLI: cold-start arch sweep on a frozen registry split.

Run (dev)::

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/tools/offline_baseline_arch_sweep.py

Optional::

    --limit 10000
    --epochs 20
    --salt baseline-v1
    --split-version baseline-v1
    --style-disagree-boost 2.0
    --style-disagree-scale 2.0
"""

from __future__ import annotations

from chess_teacher.pipelines.neural_network.offline_arch_sweep import main
from chess_teacher.pipelines.neural_network.tf_runtime import ensure_tensorflow_logging
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.process_utils import log_script_runtime_context, run_script_main

ensure_tensorflow_logging()
logger = get_logger()


if __name__ == "__main__":
    log_script_runtime_context(logger, script="offline_baseline_arch_sweep")
    run_script_main(main)
