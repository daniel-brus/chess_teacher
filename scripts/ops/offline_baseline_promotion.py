"""Offline promotion sibling: score two models on registry val (no DB promote).

Mimics ``scripts/entrypoints/baseline_promotion.py`` scoring, but uses frozen
registry val + stratified metrics. Never writes ``ml.baseline_models``.

Run (dev)::

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/ops/offline_baseline_promotion.py ^
      --candidate-uri s3://.../model.keras

Optional::

    --split-version baseline-v1
    --limit 10000
    --full-val
    --baseline-uri PATH_OR_URI
    --train-inline
    --epochs 3
    --style-disagree-boost 2.0
    --style-disagree-scale 2.0
"""

from __future__ import annotations

import argparse
import time

from chess_teacher.pipelines.neural_network.eval_metrics import (
    evaluate_datums,
    evaluate_model_uri,
    format_eval_delta,
    format_eval_metrics,
)
from chess_teacher.pipelines.neural_network.offline_eval import (
    load_registry_split,
    load_registry_val_datums,
    resolve_production_model_uri,
)
from chess_teacher.pipelines.neural_network.splits import DEFAULT_SPLIT_SALT, format_split_summary
from chess_teacher.pipelines.neural_network.tf_runtime import ensure_tensorflow_logging
from chess_teacher.pipelines.neural_network.train import BaselineTrainer
from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.process_utils import log_script_runtime_context, run_script_main

ensure_tensorflow_logging()
logger = get_logger()


def run_offline_promotion(
    *,
    split_version: str,
    limit: int,
    full_val: bool,
    candidate_uri: str | None,
    baseline_uri: str | None,
    train_inline: bool,
    epochs: int,
    style_disagree_boost: float,
    style_disagree_scale: float,
) -> int:
    if train_inline == bool(candidate_uri):
        logger.error("Pass exactly one of --candidate-uri or --train-inline.")
        return 1
    if full_val and train_inline:
        logger.error("--full-val cannot be combined with --train-inline (val sets would differ).")
        return 1

    db = get_db_client()
    try:
        production_uri = baseline_uri or resolve_production_model_uri(db)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    if train_inline:
        logger.info("Loading sample limit=%s for inline train + registry val…", limit)
        split = load_registry_split(db, limit=limit, split_version=split_version)
        print("\n" + format_split_summary(split))
        train = split.train_datums
        val = split.val_datums
        if len(train) < 30 or len(val) < 10:
            logger.error("Split too small train=%s val=%s", len(train), len(val))
            return 1
        trainer = BaselineTrainer(
            epochs=epochs,
            style_disagree_boost=style_disagree_boost,
            style_disagree_scale=style_disagree_scale,
        )
        logger.info("Training inline on train split n=%s epochs=%s…", len(train), epochs)
        t0 = time.perf_counter()
        model, train_metrics = trainer.fit(train)
        logger.info(
            "Inline fit done in %.1fs train_top1=%.4f",
            time.perf_counter() - t0,
            train_metrics.get("masked_cand_top1", 0),
        )
        candidate_metrics = evaluate_datums(model, val)
        candidate_label = f"inline@{epochs}ep"
    else:
        logger.info(
            "Loading registry val full=%s limit=%s split_version=%s…",
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
        logger.info("Scoring candidate uri=%s on val n=%s…", candidate_uri, len(val))
        assert candidate_uri is not None
        candidate_metrics = evaluate_model_uri(candidate_uri, val)
        candidate_label = "candidate"

    logger.info("Scoring baseline uri=%s on val n=%s…", production_uri, len(val))
    production_metrics = evaluate_model_uri(production_uri, val)

    print("\n=== offline promotion compare (no DB write) ===")
    print(f"split_version={split_version!r} val_n={len(val)} full_val={full_val}")
    print(f"baseline_uri={production_uri}")
    if candidate_uri:
        print(f"candidate_uri={candidate_uri}")
    print(format_eval_metrics(candidate_label, candidate_metrics))
    print(format_eval_metrics("prod", production_metrics))
    print(
        format_eval_delta(
            candidate_metrics,
            production_metrics,
            candidate_name=candidate_label,
            baseline_name="prod",
        )
    )
    print("no_promote=true  (this script never writes ml.baseline_models)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-version", type=str, default=DEFAULT_SPLIT_SALT)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument(
        "--full-val",
        action="store_true",
        help="Score all registry val games instead of a --limit sample.",
    )
    parser.add_argument("--candidate-uri", type=str, default=None)
    parser.add_argument(
        "--baseline-uri",
        type=str,
        default=None,
        help="Override production URI (default: latest production row).",
    )
    parser.add_argument(
        "--train-inline",
        action="store_true",
        help="Cold-start train on registry train split, then compare on val.",
    )
    parser.add_argument("--epochs", type=int, default=BaselineTrainer.DEFAULT_EPOCHS)
    parser.add_argument("--style-disagree-boost", type=float, default=2.0)
    parser.add_argument("--style-disagree-scale", type=float, default=2.0)
    args = parser.parse_args()
    log_script_runtime_context(logger, script="offline_baseline_promotion")
    return run_offline_promotion(
        split_version=str(args.split_version),
        limit=max(50, int(args.limit)),
        full_val=bool(args.full_val),
        candidate_uri=str(args.candidate_uri) if args.candidate_uri else None,
        baseline_uri=str(args.baseline_uri) if args.baseline_uri else None,
        train_inline=bool(args.train_inline),
        epochs=max(1, int(args.epochs)),
        style_disagree_boost=float(args.style_disagree_boost),
        style_disagree_scale=float(args.style_disagree_scale),
    )


if __name__ == "__main__":
    run_script_main(main)
