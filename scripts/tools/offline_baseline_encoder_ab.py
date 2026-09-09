"""Thin CLI: MLP vs hybrid board encoder A/B on registry val (Phase 2c).

Run (dev)::

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/tools/offline_baseline_encoder_ab.py

Optional::

    --epochs 10
    --max-rounds 10
    --full-val
    --arms hybrid|mlp|both
    --reset-queue
    --batch-limit 10000
    --style-disagree-boost 2.0

``--max-rounds 1`` = single cold sample A/B.
``--max-rounds >1`` = registry train queue (game_id + mark processed);
val packed once, scored every round.
"""

from __future__ import annotations

from chess_teacher.pipelines.neural_network.offline_encoder_ab import main
from chess_teacher.pipelines.neural_network.tf_runtime import ensure_tensorflow_logging
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.process_utils import log_script_runtime_context, run_script_main

ensure_tensorflow_logging()
logger = get_logger()


if __name__ == "__main__":
    log_script_runtime_context(logger, script="offline_baseline_encoder_ab")
    run_script_main(main)
