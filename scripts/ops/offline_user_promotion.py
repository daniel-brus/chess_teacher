"""Offline user promotion sibling: score two hybrid models on user ∩ registry val.

Does not train and does not write ``ml.baseline_models``. Compare an existing
user ``.keras`` to an existing baseline ``.keras`` (the 3a parent, or any other
local hybrid weights). Val is that account's registry val only.

Run (dev)::

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/ops/offline_user_promotion.py ^
      --account-id <uuid> ^
      --user-weights storage/tmp/phase3a/user.keras ^
      --baseline-weights storage/tmp/phase3a/hybrid_parent.keras

Optional::

    --split-version baseline-v1
    --val-limit 10000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from chess_teacher.pipelines.neural_network.board_encoder import load_hybrid_board_keras
from chess_teacher.pipelines.neural_network.eval_metrics import (
    evaluate_datums,
    format_eval_delta,
    format_eval_metrics,
)
from chess_teacher.pipelines.neural_network.offline_eval import load_account_registry_bucket_datums
from chess_teacher.pipelines.neural_network.splits import DEFAULT_SPLIT_SALT, SplitBucket
from chess_teacher.pipelines.neural_network.tf_runtime import ensure_tensorflow_logging
from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.process_utils import log_script_runtime_context, run_script_main

ensure_tensorflow_logging()
logger = get_logger()


def run_offline_user_promotion(
    *,
    account_id: str,
    split_version: str,
    user_weights: str,
    baseline_weights: str,
    val_limit: int | None,
) -> int:
    aid = account_id.strip()
    if not aid:
        logger.error("--account-id is required.")
        return 1

    user_path = Path(user_weights)
    baseline_path = Path(baseline_weights)
    if not user_path.is_file():
        logger.error("User weights not found: %s", user_path)
        return 1
    if not baseline_path.is_file():
        logger.error("Baseline weights not found: %s", baseline_path)
        return 1

    db = get_db_client()
    logger.info(
        "Loading user∩registry val account_id=%s split_version=%s val_limit=%s…",
        aid,
        split_version,
        val_limit,
    )
    val = load_account_registry_bucket_datums(
        aid,
        db,
        bucket=SplitBucket.VAL,
        split_version=split_version,
        limit=val_limit,
    )
    if not val:
        logger.error("User registry val is empty for account_id=%s", aid)
        return 1

    logger.info("Scoring user weights=%s on val n=%s…", user_path, len(val))
    user_model = load_hybrid_board_keras(user_path, compile_model=False)
    user_metrics = evaluate_datums(user_model, val)

    logger.info("Scoring baseline weights=%s on val n=%s…", baseline_path, len(val))
    baseline_model = load_hybrid_board_keras(baseline_path, compile_model=False)
    baseline_metrics = evaluate_datums(baseline_model, val)

    print("\n=== offline user promotion compare (no DB write) ===")
    print(f"account_id={aid!r} split_version={split_version!r} val_n={len(val)}")
    print(f"user_weights={user_path}")
    print(f"baseline_weights={baseline_path}")
    print(format_eval_metrics("user", user_metrics))
    print(format_eval_metrics("baseline", baseline_metrics))
    print(
        format_eval_delta(
            user_metrics,
            baseline_metrics,
            candidate_name="user",
            baseline_name="baseline",
        )
    )
    print("primary=top1_sf_disagree  agree=report_only  no_promote=true")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account-id", type=str, required=True)
    parser.add_argument("--split-version", type=str, default=DEFAULT_SPLIT_SALT)
    parser.add_argument("--user-weights", type=str, required=True)
    parser.add_argument("--baseline-weights", type=str, required=True)
    parser.add_argument(
        "--val-limit",
        type=int,
        default=None,
        help="Complete-game move cap for user∩registry-val (default: all).",
    )
    args = parser.parse_args()
    log_script_runtime_context(logger, script="offline_user_promotion")
    return run_offline_user_promotion(
        account_id=str(args.account_id),
        split_version=str(args.split_version),
        user_weights=str(args.user_weights),
        baseline_weights=str(args.baseline_weights),
        val_limit=int(args.val_limit) if args.val_limit is not None else None,
    )


if __name__ == "__main__":
    run_script_main(main)
