"""Thin CLI: E22 loss-kind A/B (offline only; not wired into Phase 3/prod).

doppler run --project chess-teacher --config dev_local -- ^
  .venv\\Scripts\\python.exe scripts/tools/offline_loss_ab.py ^
  --encoder hybrid --epochs 10 --max-rounds 3 --limit 10000 --reset-queue ^
  --loss-kinds sparse,soft,sf_mix ^
  --output-dir storage/tmp/hybrid_loss_ab_r3_ep10
"""

from __future__ import annotations

from chess_teacher.pipelines.neural_network.offline_loss_ab import main
from chess_teacher.pipelines.neural_network.tf_runtime import ensure_tensorflow_logging
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.process_utils import log_script_runtime_context, run_script_main

ensure_tensorflow_logging()
logger = get_logger()


if __name__ == "__main__":
    log_script_runtime_context(logger, script="offline_loss_ab")
    run_script_main(main)
