"""Thin CLI: E21 forced-move weight A/B on registry queue.

Run hybrid (preferred)::

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/tools/offline_forced_weight_ab.py ^
      --encoder hybrid --epochs 10 --max-rounds 10 --full-val --reset-queue ^
      --output-dir storage/tmp/hybrid_forced_ab_r10_ep10_s1.5

Control = forced off; treatment = ``--forced-scale-pawns`` (default 1.5).
``--encoder mlp|hybrid`` (default mlp for back-compat).
"""

from __future__ import annotations

from chess_teacher.pipelines.neural_network.offline_forced_weight_ab import main
from chess_teacher.pipelines.neural_network.tf_runtime import ensure_tensorflow_logging
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.process_utils import log_script_runtime_context, run_script_main

ensure_tensorflow_logging()
logger = get_logger()


if __name__ == "__main__":
    log_script_runtime_context(logger, script="offline_forced_weight_ab")
    run_script_main(main)
