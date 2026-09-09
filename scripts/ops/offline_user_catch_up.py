"""Thin CLI: offline user catch-up on personal registry train queue.

Marks ``already_processed_personal`` after a successful round. Does not write
promote / baseline rows.

Run (dev)::

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/ops/offline_user_catch_up.py ^
      --account-id ACCOUNT_ID

Optional::

    --parent-uri PATH_OR_URI
    --max-rounds 50
    --min-new-moves 50
    --batch-limit 1000
    --limit N          (ignored on catch-up; finetune: sorted game_id train cap)
    --epochs 20
    --recency-lambda 0.02
    --recency-boost 2.0
    --style-disagree-boost 2.0
    --style-disagree-scale 2.0
    --baseline-disagree-boost 4.0
    --min-train-moves 300
    --output-dir DIR
"""

from __future__ import annotations

from chess_teacher.pipelines.neural_network.offline_user import main_catch_up as main
from chess_teacher.pipelines.neural_network.tf_runtime import ensure_tensorflow_logging
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.process_utils import log_script_runtime_context, run_script_main

ensure_tensorflow_logging()
logger = get_logger()


if __name__ == "__main__":
    log_script_runtime_context(logger, script="offline_user_catch_up")
    run_script_main(main)
