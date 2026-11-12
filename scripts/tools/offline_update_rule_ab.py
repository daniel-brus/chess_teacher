"""Thin CLI: offline hybrid update-rule ABCD (read-only).

Mid decision::

    doppler run --project chess-teacher --config dev_local -- `
      .venv\\Scripts\\python.exe scripts\\tools\\offline_update_rule_ab.py `
      --train-limit 30000 --val-limit 8000 --epochs 3 `
      --output storage\\tmp\\update_rule_ab_mid.json
"""

from __future__ import annotations

import os

os.environ.setdefault("ENVIRONMENT", "AGENT")

from chess_teacher.pipelines.neural_network.offline_update_rule_ab import main
from chess_teacher.pipelines.neural_network.tf_runtime import ensure_tensorflow_logging
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.process_utils import log_script_runtime_context, run_script_main

ensure_tensorflow_logging()
logger = get_logger()


if __name__ == "__main__":
    log_script_runtime_context(logger, script="offline_update_rule_ab")
    run_script_main(main)
