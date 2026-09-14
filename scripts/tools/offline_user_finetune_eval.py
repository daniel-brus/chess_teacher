"""Thin CLI: Phase 3a offline user finetune eval (hash registry only).

doppler run --project chess-teacher --config dev_local -- ^
  .venv\\Scripts\\python.exe scripts/tools/offline_user_finetune_eval.py ^
  --account-id <uuid> --parent-weights storage/tmp/hybrid_parent.keras

Cold parent then user (after backfill for second account)::

  doppler run --project chess-teacher --config dev_local -- ^
    .venv\\Scripts\\python.exe scripts/tools/offline_user_finetune_eval.py ^
    --train-parent --parent-out storage/tmp/hybrid_parent.keras ^
    --account-id <uuid>
"""

from __future__ import annotations

from chess_teacher.pipelines.neural_network.offline_user_finetune import main
from chess_teacher.pipelines.neural_network.tf_runtime import ensure_tensorflow_logging
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.process_utils import log_script_runtime_context, run_script_main

ensure_tensorflow_logging()
logger = get_logger()


if __name__ == "__main__":
    log_script_runtime_context(logger, script="offline_user_finetune_eval")
    run_script_main(main)
