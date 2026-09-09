"""Offline catch-up sibling: queue replay on registry train, frozen val.

Mimics production catch-up *shape* (count -> fetch batch -> finetune parent ->
mark processed) but never writes TrainingState or ``ml.baseline_models``.
Val is loaded once and reused every round.

Does not import or call production train / promote / catch-up entrypoints.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from chess_teacher.pipelines.neural_network.create_training_set import TrainingDataStore
from chess_teacher.pipelines.neural_network.eval_metrics import (
    EvalMetrics,
    evaluate_packed,
    format_eval_metrics,
    pack_datums_for_eval,
)
from chess_teacher.pipelines.neural_network.mlflow_utils import MLflowTracker
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


def _keras_parent_path(parent_uri: str | None) -> Path | None:
    if not parent_uri:
        return None
    path = MLflowTracker().download_keras_weights(parent_uri)
    if path is None or not path.is_file():
        raise FileNotFoundError(f"Could not resolve parent Keras weights from uri={parent_uri!r}")
    return path


def _print_val_curve(rows: list[tuple[int, str, int, EvalMetrics]]) -> None:
    print("\n=== offline catch-up val curve (frozen val, no promote DB write) ===")
    if not rows:
        print("no rounds")
        return
    for round_i, last_id, n_train, metrics in rows:
        print(
            f"round={round_i} last_game_id={last_id} train_n={n_train} "
            f"{format_eval_metrics('val', metrics)}"
        )


def _append_curve_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def run_offline_catch_up(
    *,
    split_version: str,
    val_limit: int,
    full_val: bool,
    max_rounds: int,
    min_new_moves: int,
    batch_limit: int,
    parent_uri: str | None,
    epochs: int,
    style_disagree_boost: float,
    style_disagree_scale: float,
    output_dir: str | Path | None,
) -> int:
    db = get_db_client()
    logger.info(
        "Loading frozen registry val full=%s val_limit=%s split_version=%s (once)...",
        full_val,
        val_limit,
        split_version,
    )
    val = load_registry_val_datums(
        db,
        split_version=split_version,
        limit=None if full_val else val_limit,
        full=full_val,
    )
    if len(val) < 10:
        logger.error("Val set too small: %s moves", len(val))
        return 1
    packed_val = pack_datums_for_eval(val)
    logger.info(
        "Packed frozen val once n_input=%s kept=%s",
        packed_val.n_input,
        len(packed_val.kept_datums),
    )

    try:
        parent = _keras_parent_path(parent_uri)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    store = TrainingDataStore(db)
    registry = get_split_registry(db, split_version=split_version)
    max_rounds = max(1, int(max_rounds))
    trainer = BaselineTrainer(
        epochs=epochs,
        style_disagree_boost=style_disagree_boost,
        style_disagree_scale=style_disagree_scale,
    )

    curve: list[tuple[int, str, int, EvalMetrics]] = []
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
            n_before = store.count_unprocessed_train(split_version=split_version)
            logger.info(
                "Offline catch-up check round=%s unprocessed_train=%s min=%s batch_cap=%s",
                round_i + 1,
                n_before,
                min_new_moves,
                batch_limit,
            )
            if n_before < min_new_moves:
                logger.info(
                    "Caught up: unprocessed_train=%s < min=%s (remainder left for later).",
                    n_before,
                    min_new_moves,
                )
                _print_val_curve(curve)
                return 0

            datums, game_ids = store.fetch_unprocessed_train_batch(
                split_version=split_version,
                limit=batch_limit,
            )
            if not datums or not game_ids:
                logger.error(
                    "Pending=%s but unprocessed train batch empty.",
                    n_before,
                )
                _print_val_curve(curve)
                return 2

            round_i += 1
            last_id = game_ids[-1]
            logger.info(
                "=== offline catch-up round %s/%s train_n=%s games=%s parent=%s last_game_id=%s ===",
                round_i,
                max_rounds,
                len(datums),
                len(game_ids),
                parent,
                last_id,
            )
            t0 = time.monotonic()
            model, _metrics = trainer.fit(datums, weights_path=parent)
            save_path = root_path / f"round_{round_i}" / "model.keras"
            BaselineTrainer.save(model, save_path)
            val_metrics = evaluate_packed(model, packed_val)
            marked = registry.mark_processed(game_ids)
            if marked <= 0:
                logger.error(
                    "Fit succeeded but mark_processed updated 0 rows (games=%s) - "
                    "stop to avoid infinite loop.",
                    len(game_ids),
                )
                _print_val_curve(curve)
                return 2
            parent = save_path
            elapsed_s = time.monotonic() - t0
            curve.append((round_i, last_id, len(datums), val_metrics))
            print(format_eval_metrics(f"round{round_i}", val_metrics))

            n_after = store.count_unprocessed_train(split_version=split_version)
            _append_curve_jsonl(
                root_path / "val_curve.jsonl",
                {
                    "round": round_i,
                    "last_game_id": last_id,
                    "train_n": len(datums),
                    "games": len(game_ids),
                    "marked": marked,
                    "pending_before": n_before,
                    "pending_after": n_after,
                    "elapsed_s": round(elapsed_s, 1),
                    "metrics": val_metrics.as_dict(),
                },
            )
            if n_after >= n_before:
                logger.error(
                    "Train succeeded but unprocessed count did not drop "
                    "(before=%s after=%s) - stop to avoid infinite loop.",
                    n_before,
                    n_after,
                )
                _print_val_curve(curve)
                return 2

        logger.error("Hit max_rounds=%s with unprocessed still above min - stopping.", max_rounds)
        _print_val_curve(curve)
        return 3


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-version", type=str, default=DEFAULT_SPLIT_SALT)
    parser.add_argument(
        "--val-limit",
        "--limit",
        dest="val_limit",
        type=int,
        default=10000,
        help=(
            "Val SAMPLE size when not --full-val (oldest-N then val slice). "
            "Does not cap training batches; see --batch-limit. Ignored if --full-val."
        ),
    )
    parser.add_argument(
        "--full-val",
        action="store_true",
        help="Score all registry val games (official frozen val; ignores --val-limit).",
    )
    parser.add_argument("--max-rounds", type=int, default=50)
    parser.add_argument("--min-new-moves", type=int, default=MIN_NEW_MOVES_BASELINE)
    parser.add_argument("--batch-limit", type=int, default=MAX_MOVES_PER_BASELINE_BATCH)
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
    return run_offline_catch_up(
        split_version=str(args.split_version),
        val_limit=max(50, int(args.val_limit)),
        full_val=bool(args.full_val),
        max_rounds=max(1, int(args.max_rounds)),
        min_new_moves=max(1, int(args.min_new_moves)),
        batch_limit=max(1, int(args.batch_limit)),
        parent_uri=str(args.parent_uri) if args.parent_uri else None,
        epochs=max(1, int(args.epochs)),
        style_disagree_boost=float(args.style_disagree_boost),
        style_disagree_scale=float(args.style_disagree_scale),
        output_dir=str(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    from chess_teacher.utils.process_utils import run_script_main

    run_script_main(main)
