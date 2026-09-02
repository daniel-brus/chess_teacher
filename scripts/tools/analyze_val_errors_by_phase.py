"""Thin CLI: registry-val errors by game phase (E10-E11). Prints text only.

Load a model URI + frozen registry val, score overall + opening/middle/endgame
slices, then shortlist endgame + SF-disagree + top1-miss rows (FEN, ply, game_id).

Does not bump feat versions or train.

Run (dev)::

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/tools/analyze_val_errors_by_phase.py ^
      --model-uri PATH_OR_URI

Optional::

    --split-version baseline-v1
    --limit 10000
    --full-val
    --error-limit 30
"""

from __future__ import annotations

import argparse

from chess_teacher.pipelines.neural_network.eval_metrics import format_phase_error_report
from chess_teacher.pipelines.neural_network.offline_eval import load_registry_val_datums
from chess_teacher.pipelines.neural_network.splits import DEFAULT_SPLIT_SALT
from chess_teacher.pipelines.neural_network.tf_runtime import ensure_tensorflow_logging
from chess_teacher.pipelines.neural_network.train import load_candidate_style_from_uri
from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.process_utils import log_script_runtime_context, run_script_main

ensure_tensorflow_logging()
logger = get_logger()


def run_analyze_val_errors_by_phase(
    *,
    model_uri: str,
    split_version: str,
    limit: int,
    full_val: bool,
    error_limit: int,
) -> int:
    db = get_db_client()
    logger.info(
        "Loading frozen registry val full=%s limit=%s split_version=%s…",
        full_val,
        limit,
        split_version,
    )
    val = load_registry_val_datums(
        db,
        split_version=split_version,
        limit=None if full_val else limit,
        full=full_val,
    )
    if len(val) < 10:
        logger.error("Val set too small: %s moves", len(val))
        return 1
    logger.info("Loading model uri=%s…", model_uri)
    model = load_candidate_style_from_uri(model_uri)
    print("\n=== val errors by phase (text only, no feat v4) ===")
    print(f"split_version={split_version!r} val_n={len(val)} full_val={full_val}")
    print(f"model_uri={model_uri}")
    print(format_phase_error_report(model, val, error_limit=error_limit))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-uri", type=str, required=True)
    parser.add_argument("--split-version", type=str, default=DEFAULT_SPLIT_SALT)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--full-val", action="store_true")
    parser.add_argument("--error-limit", type=int, default=30)
    args = parser.parse_args()
    log_script_runtime_context(logger, script="analyze_val_errors_by_phase")
    return run_analyze_val_errors_by_phase(
        model_uri=str(args.model_uri),
        split_version=str(args.split_version),
        limit=max(50, int(args.limit)),
        full_val=bool(args.full_val),
        error_limit=max(1, int(args.error_limit)),
    )


if __name__ == "__main__":
    run_script_main(main)
