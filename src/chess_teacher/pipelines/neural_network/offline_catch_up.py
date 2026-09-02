"""Offline catch-up sibling: incremental replay on registry train, frozen val.

Mimics production catch-up *shape* (count -> fetch batch -> finetune parent ->
advance cutoff) but stays split-hygienic and never writes TrainingState or
``ml.baseline_models``. Val is loaded once and reused every round.

Does not import or call production train / promote / catch-up entrypoints.
"""

from __future__ import annotations

import argparse
import tempfile
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

from chess_teacher.pipelines.neural_network.create_training_set import TrainingDataStore
from chess_teacher.pipelines.neural_network.eval_metrics import (
    EvalMetrics,
    evaluate_datums,
    format_eval_metrics,
)
from chess_teacher.pipelines.neural_network.mlflow_utils import MLflowTracker
from chess_teacher.pipelines.neural_network.models import TrainingState
from chess_teacher.pipelines.neural_network.offline_eval import load_registry_val_datums
from chess_teacher.pipelines.neural_network.pipeline_steps import (
    MAX_MOVES_PER_BASELINE_BATCH,
    MIN_NEW_MOVES_BASELINE,
)
from chess_teacher.pipelines.neural_network.split_registry import get_split_registry
from chess_teacher.pipelines.neural_network.splits import DEFAULT_SPLIT_SALT
from chess_teacher.pipelines.neural_network.train import BaselineTrainer
from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.logging import get_logger

logger = get_logger()


def _parse_start_cutoff(raw: str) -> datetime:
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def _keras_parent_path(parent_uri: str | None) -> Path | None:
    if not parent_uri:
        return None
    path = MLflowTracker().download_keras_weights(parent_uri)
    if path is None or not path.is_file():
        raise FileNotFoundError(f"Could not resolve parent Keras weights from uri={parent_uri!r}")
    return path


def _print_val_curve(rows: list[tuple[int, datetime | None, int, EvalMetrics]]) -> None:
    print("\n=== offline catch-up val curve (frozen val, no DB write) ===")
    if not rows:
        print("no rounds")
        return
    for round_i, cutoff, n_train, metrics in rows:
        print(
            f"round={round_i} cutoff={cutoff} train_n={n_train} {format_eval_metrics('val', metrics)}"
        )


def run_offline_catch_up(
    *,
    split_version: str,
    limit: int,
    full_val: bool,
    max_rounds: int,
    min_new_moves: int,
    batch_limit: int,
    start_cutoff: datetime | None,
    start_from_production_cutoff: bool,
    parent_uri: str | None,
    epochs: int,
    style_disagree_boost: float,
    style_disagree_scale: float,
    output_dir: str | Path | None,
) -> int:
    if start_cutoff is not None and start_from_production_cutoff:
        logger.error("Pass only one of --start-cutoff or --start-from-production-cutoff.")
        return 1

    db = get_db_client()
    logger.info(
        "Loading frozen registry val full=%s limit=%s split_version=%s (once)…",
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

    cutoff: datetime | None = start_cutoff
    if start_from_production_cutoff:
        state = TrainingState.for_baseline(db)
        cutoff = state.last_trained_data_cutoff
        logger.info("Read-only production cutoff=%s (TrainingState not written)", cutoff)

    try:
        parent = _keras_parent_path(parent_uri)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    store = TrainingDataStore(db)
    registry = get_split_registry(db, split_version=split_version)
    exclude_sql = registry.exclude_holdout_games_sql()
    max_rounds = max(1, int(max_rounds))
    trainer = BaselineTrainer(
        epochs=epochs,
        style_disagree_boost=style_disagree_boost,
        style_disagree_scale=style_disagree_scale,
    )

    curve: list[tuple[int, datetime | None, int, EvalMetrics]] = []
    out_ctx: Any
    if output_dir is not None:
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        out_ctx = nullcontext(str(out_root))
    else:
        out_ctx = tempfile.TemporaryDirectory(prefix="offline_catch_up_")

    with out_ctx as root:
        root_path = Path(root)
        round_i = 0
        while round_i < max_rounds:
            n_before = store.count_since(cutoff, extra_where=exclude_sql)
            logger.info(
                "Offline catch-up check round=%s eligible=%s cutoff=%s min=%s batch_cap=%s",
                round_i + 1,
                n_before,
                cutoff,
                min_new_moves,
                batch_limit,
            )
            if n_before < min_new_moves:
                logger.info(
                    "Caught up: eligible=%s < min=%s (remainder left for later).",
                    n_before,
                    min_new_moves,
                )
                _print_val_curve(curve)
                return 0

            datums, max_t = store.fetch_since(
                cutoff,
                limit=batch_limit,
                extra_where=exclude_sql,
            )
            split = registry.split_datums(datums, assign_if_missing=True)
            train = split.train_datums
            if not train:
                logger.error("No train datums after registry split (assign_if_missing=True).")
                _print_val_curve(curve)
                return 1

            round_i += 1
            logger.info(
                "=== offline catch-up round %s/%s train_n=%s parent=%s ===",
                round_i,
                max_rounds,
                len(train),
                parent,
            )
            model, _metrics = trainer.fit(train, weights_path=parent)
            val_metrics = evaluate_datums(model, val)
            save_path = root_path / f"round_{round_i}" / "model.keras"
            BaselineTrainer.save(model, save_path)
            parent = save_path
            curve.append((round_i, max_t, len(train), val_metrics))
            print(format_eval_metrics(f"round{round_i}", val_metrics))

            n_after = store.count_since(max_t, extra_where=exclude_sql)
            if n_after >= n_before and max_t == cutoff:
                logger.error(
                    "Train succeeded but eligible count did not drop "
                    "(before=%s after=%s cutoff=%s) - stop to avoid infinite loop.",
                    n_before,
                    n_after,
                    cutoff,
                )
                _print_val_curve(curve)
                return 2
            cutoff = max_t

        logger.error("Hit max_rounds=%s with eligible still above min - stopping.", max_rounds)
        _print_val_curve(curve)
        return 3


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-version", type=str, default=DEFAULT_SPLIT_SALT)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument(
        "--full-val",
        action="store_true",
        help="Score all registry val games (official frozen val).",
    )
    parser.add_argument("--max-rounds", type=int, default=50)
    parser.add_argument("--min-new-moves", type=int, default=MIN_NEW_MOVES_BASELINE)
    parser.add_argument("--batch-limit", type=int, default=MAX_MOVES_PER_BASELINE_BATCH)
    cutoff_group = parser.add_mutually_exclusive_group()
    cutoff_group.add_argument(
        "--start-cutoff",
        type=str,
        default=None,
        help="ISO datetime; exclusive games.end_time lower bound.",
    )
    cutoff_group.add_argument(
        "--start-from-production-cutoff",
        action="store_true",
        help="Read TrainingState.for_baseline cutoff only (no write).",
    )
    parser.add_argument(
        "--parent-uri",
        type=str,
        default=None,
        help="Optional parent Keras / MLflow URI (cold-start round 1 if omitted).",
    )
    parser.add_argument("--epochs", type=int, default=BaselineTrainer.DEFAULT_EPOCHS)
    parser.add_argument("--style-disagree-boost", type=float, default=2.0)
    parser.add_argument("--style-disagree-scale", type=float, default=2.0)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    start_cutoff: datetime | None = None
    if args.start_cutoff:
        try:
            start_cutoff = _parse_start_cutoff(str(args.start_cutoff))
        except ValueError:
            logger.error("Invalid --start-cutoff %r (use ISO datetime).", args.start_cutoff)
            return 1
    return run_offline_catch_up(
        split_version=str(args.split_version),
        limit=max(50, int(args.limit)),
        full_val=bool(args.full_val),
        max_rounds=max(1, int(args.max_rounds)),
        min_new_moves=max(1, int(args.min_new_moves)),
        batch_limit=max(1, int(args.batch_limit)),
        start_cutoff=start_cutoff,
        start_from_production_cutoff=bool(args.start_from_production_cutoff),
        parent_uri=str(args.parent_uri) if args.parent_uri else None,
        epochs=max(1, int(args.epochs)),
        style_disagree_boost=float(args.style_disagree_boost),
        style_disagree_scale=float(args.style_disagree_scale),
        output_dir=str(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    from chess_teacher.utils.process_utils import run_script_main

    run_script_main(main)
