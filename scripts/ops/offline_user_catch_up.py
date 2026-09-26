"""Offline user catch-up: replay one account's unprocessed registry-train.

Progress is ``already_processed_personal`` on registry-train games, not
``end_time``. Each round finetunes from the parent (then the previous round)
and re-scores that account's registry val. Does not write ``TrainingState``
or ``ml.baseline_models``.

Under ``--min-new-moves`` (default 300) the script skips finetune and leaves
the baseline in place.

Run (dev)::

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/ops/offline_user_catch_up.py ^
      --account-id <uuid> ^
      --parent-weights storage/tmp/phase3a/hybrid_parent.keras

Optional::

    --split-version baseline-v1
    --val-limit 10000
    --max-rounds 50
    --min-new-moves 300
    --batch-limit 10000
    --epochs 20
    --style-disagree-boost 4.0
    --output-dir DIR
"""

from __future__ import annotations

import argparse
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from chess_teacher.pipelines.neural_network.board_encoder import (
    HybridBoardTrainer,
    load_hybrid_board_keras,
)
from chess_teacher.pipelines.neural_network.create_training_set import TrainingDataStore
from chess_teacher.pipelines.neural_network.eval_metrics import (
    EvalMetrics,
    evaluate_datums,
    format_eval_delta,
    format_eval_metrics,
)
from chess_teacher.pipelines.neural_network.models import PROCESSED_FLAG_PERSONAL
from chess_teacher.pipelines.neural_network.offline_eval import (
    account_registry_extra_where,
    load_account_registry_bucket_datums,
)
from chess_teacher.pipelines.neural_network.offline_user_finetune import (
    DEFAULT_USER_STYLE_DISAGREE_BOOST,
    MIN_USER_TRAIN_MOVES,
)
from chess_teacher.pipelines.neural_network.pipeline_steps import MAX_MOVES_PER_BASELINE_BATCH
from chess_teacher.pipelines.neural_network.split_registry import get_split_registry
from chess_teacher.pipelines.neural_network.splits import DEFAULT_SPLIT_SALT, SplitBucket
from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.process_utils import log_script_runtime_context, run_script_main

logger = get_logger()


def _print_curve(
    rows: list[tuple[int, int, EvalMetrics]],
    *,
    baseline: EvalMetrics | None,
) -> None:
    print("\n=== offline user catch-up val curve (user∩registry val) ===")
    if baseline is not None:
        print(format_eval_metrics("baseline", baseline))
    if not rows:
        print("no rounds")
        return
    for round_i, n_train, metrics in rows:
        print(f"round={round_i} train_n={n_train} {format_eval_metrics('user', metrics)}")
        if baseline is not None:
            print(
                format_eval_delta(
                    metrics,
                    baseline,
                    candidate_name=f"user_r{round_i}",
                    baseline_name="baseline",
                )
            )


def run_offline_user_catch_up(
    *,
    account_id: str,
    split_version: str,
    parent_weights: str,
    val_limit: int | None,
    max_rounds: int,
    min_new_moves: int,
    batch_limit: int,
    epochs: int,
    style_disagree_boost: float,
    style_disagree_scale: float,
    output_dir: str | None,
) -> int:
    aid = account_id.strip()
    if not aid:
        logger.error("--account-id is required.")
        return 1

    parent_path = Path(parent_weights)
    if not parent_path.is_file():
        logger.error("Parent weights not found: %s", parent_path)
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
    if len(val) < 10:
        logger.error("User registry val too small: %s moves for account_id=%s", len(val), aid)
        return 1

    extra_where = account_registry_extra_where(aid)
    store = TrainingDataStore(db)
    registry = get_split_registry(db, split_version=split_version)

    curve: list[tuple[int, int, EvalMetrics]] = []
    baseline_metrics: EvalMetrics | None = None
    parent: Path = parent_path
    out_ctx: Any
    if output_dir is not None:
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        out_ctx = nullcontext(str(out_root))
    else:
        out_ctx = tempfile.TemporaryDirectory(prefix="offline_user_catch_up_")

    with out_ctx as root:
        root_path = Path(root)
        round_i = 0
        while round_i < max_rounds:
            n_before = store.count_unprocessed_train(
                split_version=split_version,
                flag_column=PROCESSED_FLAG_PERSONAL,
                extra_where=extra_where,
            )
            logger.info(
                "User catch-up check account=%s round=%s unprocessed=%s min=%s",
                aid,
                round_i + 1,
                n_before,
                min_new_moves,
            )
            if n_before < min_new_moves:
                if round_i == 0:
                    logger.info(
                        "User registry-train below min-data gate: n=%s < %s — "
                        "skip finetune / serve baseline.",
                        n_before,
                        min_new_moves,
                    )
                else:
                    logger.info(
                        "Caught up: unprocessed=%s < min=%s (remainder left).",
                        n_before,
                        min_new_moves,
                    )
                _print_curve(curve, baseline=baseline_metrics)
                print(
                    f"account_id={aid!r} split_version={split_version!r} "
                    f"rounds={round_i} pending={n_before} "
                    "flag=already_processed_personal "
                    "no_training_state_write=true no_baseline_models_write=true "
                    "primary=top1_sf_disagree"
                )
                return 0

            datums, game_ids = store.fetch_unprocessed_train_batch(
                split_version=split_version,
                limit=batch_limit,
                flag_column=PROCESSED_FLAG_PERSONAL,
                extra_where=extra_where,
            )
            if not datums or not game_ids:
                logger.error("Empty unprocessed batch with pending=%s", n_before)
                _print_curve(curve, baseline=baseline_metrics)
                return 2

            if baseline_metrics is None:
                logger.info("Scoring parent weights=%s on val n=%s…", parent_path, len(val))
                baseline_model = load_hybrid_board_keras(parent_path, compile_model=False)
                baseline_metrics = evaluate_datums(baseline_model, val)
                print(format_eval_metrics("baseline", baseline_metrics))

            round_i += 1
            logger.info(
                "=== user catch-up round %s/%s train_n=%s games=%s parent=%s ===",
                round_i,
                max_rounds,
                len(datums),
                len(game_ids),
                parent,
            )
            trainer = HybridBoardTrainer(
                epochs=epochs,
                style_disagree_boost=style_disagree_boost,
                style_disagree_scale=style_disagree_scale,
            )
            model, _train_metrics = trainer.fit(
                datums,
                weights_path=parent,
                require_parent_weights=True,
            )
            user_metrics = evaluate_datums(model, val)
            save_path = root_path / f"round_{round_i}" / "model.keras"
            HybridBoardTrainer.save(model, save_path)
            print(format_eval_metrics(f"round{round_i}", user_metrics))
            if baseline_metrics is not None:
                print(
                    format_eval_delta(
                        user_metrics,
                        baseline_metrics,
                        candidate_name=f"user_r{round_i}",
                        baseline_name="baseline",
                    )
                )

            marked = registry.mark_processed(
                game_ids,
                flag_column=PROCESSED_FLAG_PERSONAL,
            )
            if marked <= 0:
                logger.error("mark_processed updated 0 personal rows")
                _print_curve(curve, baseline=baseline_metrics)
                return 2

            n_after = store.count_unprocessed_train(
                split_version=split_version,
                flag_column=PROCESSED_FLAG_PERSONAL,
                extra_where=extra_where,
            )
            if n_after >= n_before:
                logger.error(
                    "Unprocessed did not drop (before=%s after=%s)",
                    n_before,
                    n_after,
                )
                _print_curve(curve, baseline=baseline_metrics)
                return 2

            parent = save_path
            curve.append((round_i, len(datums), user_metrics))

        n_left = store.count_unprocessed_train(
            split_version=split_version,
            flag_column=PROCESSED_FLAG_PERSONAL,
            extra_where=extra_where,
        )
        logger.error(
            "Hit max_rounds=%s with unprocessed=%s still at or above min=%s.",
            max_rounds,
            n_left,
            min_new_moves,
        )
        _print_curve(curve, baseline=baseline_metrics)
        return 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account-id", type=str, required=True)
    parser.add_argument("--split-version", type=str, default=DEFAULT_SPLIT_SALT)
    parser.add_argument("--parent-weights", type=str, required=True)
    parser.add_argument(
        "--val-limit",
        type=int,
        default=None,
        help="Complete-game move cap for user∩registry-val (default: all).",
    )
    parser.add_argument("--max-rounds", type=int, default=50)
    parser.add_argument("--min-new-moves", type=int, default=MIN_USER_TRAIN_MOVES)
    parser.add_argument("--batch-limit", type=int, default=MAX_MOVES_PER_BASELINE_BATCH)
    parser.add_argument("--epochs", type=int, default=HybridBoardTrainer.DEFAULT_EPOCHS)
    parser.add_argument(
        "--style-disagree-boost",
        type=float,
        default=DEFAULT_USER_STYLE_DISAGREE_BOOST,
    )
    parser.add_argument("--style-disagree-scale", type=float, default=2.0)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    log_script_runtime_context(logger, script="offline_user_catch_up")
    return run_offline_user_catch_up(
        account_id=str(args.account_id),
        split_version=str(args.split_version),
        parent_weights=str(args.parent_weights),
        val_limit=int(args.val_limit) if args.val_limit is not None else None,
        max_rounds=max(1, int(args.max_rounds)),
        min_new_moves=max(1, int(args.min_new_moves)),
        batch_limit=max(1, int(args.batch_limit)),
        epochs=max(1, int(args.epochs)),
        style_disagree_boost=float(args.style_disagree_boost),
        style_disagree_scale=float(args.style_disagree_scale),
        output_dir=str(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    run_script_main(main)
