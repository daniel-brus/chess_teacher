"""Offline baseline train + game-level val/test eval (Phase 1 / 1b).

Loads a fixed sample from Postgres, assigns splits via **persistent registry**
(``ml.game_split_assignments``), trains on train only, reports stratified val metrics.
Does **not** touch production baseline pipelines.

Run backfill once per environment (or rely on assign-on-read during this script)::

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/tools/backfill_game_splits.py

Then train+eval::

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/tools/offline_baseline_train_eval.py

Optional::

    --limit 2000
    --epochs 8
    --salt baseline-v1
    --eval-test
    --style-disagree-boost 2.0
    --style-disagree-scale 2.0
"""

from __future__ import annotations

import argparse
import time

from chess_teacher.pipelines.neural_network.eval_metrics import (
    evaluate_datums,
    format_eval_metrics,
)
from chess_teacher.pipelines.neural_network.offline_eval import load_registry_split
from chess_teacher.pipelines.neural_network.splits import DEFAULT_SPLIT_SALT, format_split_summary
from chess_teacher.pipelines.neural_network.train import BaselineTrainer
from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.process_utils import log_script_runtime_context, run_script_main

logger = get_logger()


def run_offline_train_eval(
    *,
    limit: int,
    epochs: int,
    salt: str,
    eval_test: bool,
    style_disagree_boost: float,
    style_disagree_scale: float,
) -> int:
    db = get_db_client()
    logger.info("Loading datums limit=%s (cutoff=None for offline sample)…", limit)
    split = load_registry_split(db, limit=limit, split_version=salt)
    n_datums = sum(c.n_moves for c in split.counts)
    if n_datums < 50:
        logger.error("Need more datums; got %s", n_datums)
        return 1

    print("\n" + format_split_summary(split))

    train = split.train_datums
    val = split.val_datums
    if len(train) < 30:
        logger.error("Train split too small: %s moves", len(train))
        return 1
    if len(val) < 10:
        logger.error("Val split too small: %s moves", len(val))
        return 1

    trainer = BaselineTrainer(
        epochs=epochs,
        style_disagree_boost=style_disagree_boost,
        style_disagree_scale=style_disagree_scale,
    )
    logger.info(
        "Training on train split n=%s epochs=%s style_disagree_boost=%s…",
        len(train),
        epochs,
        style_disagree_boost,
    )
    t0 = time.perf_counter()
    model, train_metrics = trainer.fit(train)
    fit_s = time.perf_counter() - t0
    logger.info("Train fit done in %.1fs train_top1=%.4f", fit_s, train_metrics.get("masked_cand_top1", 0))

    print("\n=== eval metrics (stratified) ===")
    val_metrics = evaluate_datums(model, val)
    print(format_eval_metrics("val", val_metrics))

    if eval_test and split.test_datums:
        test_metrics = evaluate_datums(model, split.test_datums)
        print(format_eval_metrics("test", test_metrics))
    elif eval_test:
        print("test  (skipped — empty test split for this sample)")

    print(f"\nfit_s={fit_s:.1f} train_n={len(train)} val_n={len(val)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=BaselineTrainer.DEFAULT_EPOCHS)
    parser.add_argument("--salt", type=str, default=DEFAULT_SPLIT_SALT)
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="Also score the held-out test split (use sparingly).",
    )
    parser.add_argument("--style-disagree-boost", type=float, default=2.0)
    parser.add_argument("--style-disagree-scale", type=float, default=2.0)
    args = parser.parse_args()
    log_script_runtime_context(logger, script="offline_baseline_train_eval")
    return run_offline_train_eval(
        limit=max(50, args.limit),
        epochs=max(1, args.epochs),
        salt=args.salt,
        eval_test=bool(args.eval_test),
        style_disagree_boost=float(args.style_disagree_boost),
        style_disagree_scale=float(args.style_disagree_scale),
    )


if __name__ == "__main__":
    run_script_main(main)
