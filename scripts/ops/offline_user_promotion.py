"""Thin CLI: offline user promotion compare on frozen hash-split val (no DB write).

Run (dev)::

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/ops/offline_user_promotion.py ^
      --account-id ACCOUNT_ID --train-inline

Optional::

    --candidate-uri PATH_OR_URI
    --parent-uri PATH_OR_URI
    --baseline-uri PATH_OR_URI
    --limit N          (train game-id cap BEFORE hydrate; oldest complete games)
    --epochs 20
    --recency-lambda 0.02
    --recency-boost 2.0
    --style-disagree-boost 2.0
    --style-disagree-scale 2.0
    --baseline-disagree-boost 4.0
    --min-train-moves 300
"""

from __future__ import annotations

from chess_teacher.pipelines.neural_network.offline_user import main_promotion as main
from chess_teacher.pipelines.neural_network.tf_runtime import ensure_tensorflow_logging
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.process_utils import log_script_runtime_context, run_script_main

ensure_tensorflow_logging()
logger = get_logger()


if __name__ == "__main__":
    log_script_runtime_context(logger, script="offline_user_promotion")
    run_script_main(main)
