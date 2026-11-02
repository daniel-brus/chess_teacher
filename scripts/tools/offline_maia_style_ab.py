"""Thin CLI: offline hybrid vs spatial-policy A/B (read-only).

Cloud smoke::

    .venv/bin/python scripts/tools/offline_maia_style_ab.py \\
      --train-limit 2000 --val-limit 500 --epochs 2 \\
      --output storage/tmp/maia_style_ab_smoke.json

Local decision run (full train and val, test unread)::

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/tools/offline_maia_style_ab.py ^
      --full-buckets --epochs 2 --output storage/tmp/maia_style_ab_full.json
"""

from __future__ import annotations

import os

os.environ.setdefault("ENVIRONMENT", "AGENT")

from chess_teacher.pipelines.neural_network.offline_maia_style_ab import main
from chess_teacher.pipelines.neural_network.tf_runtime import ensure_tensorflow_logging
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.process_utils import log_script_runtime_context, run_script_main

ensure_tensorflow_logging()
logger = get_logger()


if __name__ == "__main__":
    log_script_runtime_context(logger, script="offline_maia_style_ab")
    run_script_main(main)
