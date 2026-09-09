"""Offline encoder A/B: flat-state MLP vs hybrid board+state (Phase 2c / E20).

Train/catch-up uses the **registry train queue** (unprocessed train games in
``game_id`` order; mark ``already_processed_baseline`` after successful fit).
See ``.agents/docs/ml-train-queue.md``. No end_time cutoff.

Val is packed once and reused every round (board + state + move_feats).
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from chess_teacher.pipelines.neural_network.board_encoder import HybridBoardTrainer
from chess_teacher.pipelines.neural_network.board_tensor import (
    BOARD_TENSOR_CHANNELS,
    BOARD_TENSOR_VERSION,
)
from chess_teacher.pipelines.neural_network.candidate_eval import (
    CANDIDATE_MOVE_FEAT_VERSION,
    MOVE_FEAT_DIM,
)
from chess_teacher.pipelines.neural_network.create_training_set import TrainingDataStore
from chess_teacher.pipelines.neural_network.eval_metrics import (
    EvalMetrics,
    evaluate_packed,
    format_eval_delta,
    format_eval_metrics,
    pack_datums_for_eval,
)
from chess_teacher.pipelines.neural_network.offline_eval import (
    load_registry_prefix_split,
    load_registry_val_datums,
)
from chess_teacher.pipelines.neural_network.pipeline_steps import (
    MAX_MOVES_PER_BASELINE_BATCH,
    MIN_NEW_MOVES_BASELINE,
)
from chess_teacher.pipelines.neural_network.split_registry import (
    PROCESSED_FLAG_BASELINE,
    get_split_registry,
)
from chess_teacher.pipelines.neural_network.splits import DEFAULT_SPLIT_SALT, format_split_summary
from chess_teacher.pipelines.neural_network.train import BaselineTrainer
from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.logging import get_logger

logger = get_logger()


def _keras_param_count(model: Any) -> int:
    fn = getattr(model, "count_params", None)
    if callable(fn):
        return int(fn())
    return 0


def _print_arm(name: str, metrics: EvalMetrics, n_params: int) -> None:
    print(
        format_eval_metrics(name, metrics)
        + f" params={n_params} feat_version={CANDIDATE_MOVE_FEAT_VERSION} "
        f"feat_dim={MOVE_FEAT_DIM} board_version={BOARD_TENSOR_VERSION} "
        f"board_C={BOARD_TENSOR_CHANNELS}"
    )


def _append_curve_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def run_encoder_ab_single(
    *,
    limit: int,
    epochs: int,
    split_version: str,
    style_disagree_boost: float,
    style_disagree_scale: float,
    arms: str = "both",
    full_val: bool = False,
) -> int:
    """One cold-start round on sample train (+ optional full registry val)."""
    db = get_db_client()
    logger.info(
        "Encoder A/B (single) load registry split limit=%s split_version=%s "
        "arms=%s full_val=%s...",
        limit,
        split_version,
        arms,
        full_val,
    )
    split = load_registry_prefix_split(
        db,
        limit=limit,
        split_version=split_version,
    )
    print("\n" + format_split_summary(split))
    train = split.train_datums
    if full_val:
        val = load_registry_val_datums(
            db,
            split_version=split_version,
            full=True,
        )
        print(f"\nfull registry val moves={len(val)}")
    else:
        val = split.val_datums
    if len(train) < 30 or len(val) < 10:
        logger.error("Train/val too small train=%s val=%s", len(train), len(val))
        return 1

    packed_val = pack_datums_for_eval(val)
    mlp_metrics: EvalMetrics | None = None
    hybrid_metrics: EvalMetrics | None = None
    mlp_params = 0
    hybrid_params = 0

    if arms in ("both", "mlp"):
        trainer = BaselineTrainer(
            epochs=epochs,
            style_disagree_boost=style_disagree_boost,
            style_disagree_scale=style_disagree_scale,
        )
        model, _ = trainer.fit(train, weights_path=None)
        mlp_params = _keras_param_count(model)
        mlp_metrics = evaluate_packed(model, packed_val)
        _print_arm("mlp", mlp_metrics, mlp_params)

    if arms in ("both", "hybrid"):
        trainer_h = HybridBoardTrainer(
            epochs=epochs,
            style_disagree_boost=style_disagree_boost,
            style_disagree_scale=style_disagree_scale,
        )
        model_h, _ = trainer_h.fit(train, weights_path=None)
        hybrid_params = _keras_param_count(model_h)
        hybrid_metrics = evaluate_packed(model_h, packed_val)
        _print_arm("hybrid", hybrid_metrics, hybrid_params)

    print("\n=== encoder A/B (change family = encoder only) ===")
    if mlp_metrics is not None and hybrid_metrics is not None:
        print(
            format_eval_delta(
                hybrid_metrics, mlp_metrics, candidate_name="hybrid", baseline_name="mlp"
            )
        )
    return 0


def run_encoder_queue_catch_up(
    *,
    split_version: str,
    full_val: bool,
    val_limit: int,
    max_rounds: int,
    min_new_moves: int,
    batch_limit: int,
    epochs: int,
    style_disagree_boost: float,
    style_disagree_scale: float,
    arms: str,
    output_dir: str | Path | None,
    reset_queue: bool,
    parent_hybrid: Path | None,
    parent_mlp: Path | None,
) -> int:
    """Registry-queue catch-up A/B: pack val once; mark train games after fit."""
    db = get_db_client()
    store = TrainingDataStore(db)
    registry = get_split_registry(db, split_version=split_version)

    if reset_queue:
        cleared = registry.clear_processed(flag_column=PROCESSED_FLAG_BASELINE)
        logger.warning(
            "Reset baseline train queue: clear_processed updated=%s split_version=%s",
            cleared,
            split_version,
        )

    logger.info(
        "Encoder queue A/B: pack frozen val once full=%s val_limit=%s "
        "max_rounds=%s epochs=%s arms=%s...",
        full_val,
        val_limit,
        max_rounds,
        epochs,
        arms,
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
    print(
        f"\npacked val n_input={packed_val.n_input} kept={len(packed_val.kept_datums)} "
        f"({'full registry' if full_val else 'sample'})"
    )

    max_rounds = max(1, int(max_rounds))
    mlp_trainer = BaselineTrainer(
        epochs=epochs,
        style_disagree_boost=style_disagree_boost,
        style_disagree_scale=style_disagree_scale,
    )
    hybrid_trainer = HybridBoardTrainer(
        epochs=epochs,
        style_disagree_boost=style_disagree_boost,
        style_disagree_scale=style_disagree_scale,
    )

    out_ctx: Any
    if output_dir is not None:
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        out_ctx = nullcontext(str(out_root))
    else:
        out_ctx = tempfile.TemporaryDirectory(prefix="offline_encoder_ab_")

    curve: list[tuple[int, str, int, EvalMetrics | None, EvalMetrics | None]] = []

    with out_ctx as root:
        root_path = Path(root)
        mlp_parent: Path | None = parent_mlp
        hybrid_parent: Path | None = parent_hybrid
        mlp_params = 0
        hybrid_params = 0
        round_i = 0

        while round_i < max_rounds:
            n_before = store.count_unprocessed_train(split_version=split_version)
            logger.info(
                "Encoder queue check round=%s unprocessed_train=%s min=%s batch_cap=%s",
                round_i + 1,
                n_before,
                min_new_moves,
                batch_limit,
            )
            if n_before < min_new_moves:
                logger.info(
                    "Caught up: unprocessed=%s < min=%s",
                    n_before,
                    min_new_moves,
                )
                break

            datums, game_ids = store.fetch_unprocessed_train_batch(
                split_version=split_version,
                limit=batch_limit,
            )
            if not datums or not game_ids:
                logger.error("Pending=%s but unprocessed train batch empty", n_before)
                return 2

            round_i += 1
            last_id = game_ids[-1]
            print(
                f"\n=== round {round_i}/{max_rounds} train_n={len(datums)} "
                f"games={len(game_ids)} last_game_id={last_id} ==="
            )
            mlp_metrics: EvalMetrics | None = None
            hybrid_metrics: EvalMetrics | None = None
            t0 = time.monotonic()

            if arms in ("both", "mlp"):
                model, _ = mlp_trainer.fit(datums, weights_path=mlp_parent)
                mlp_params = _keras_param_count(model)
                mlp_metrics = evaluate_packed(model, packed_val)
                save_mlp = root_path / f"round_{round_i}" / "mlp.keras"
                BaselineTrainer.save(model, save_mlp)
                mlp_parent = save_mlp
                _print_arm(f"mlp_r{round_i}", mlp_metrics, mlp_params)

            if arms in ("both", "hybrid"):
                model_h, _ = hybrid_trainer.fit(datums, weights_path=hybrid_parent)
                hybrid_params = _keras_param_count(model_h)
                hybrid_metrics = evaluate_packed(model_h, packed_val)
                save_h = root_path / f"round_{round_i}" / "hybrid.keras"
                HybridBoardTrainer.save(model_h, save_h)
                hybrid_parent = save_h
                _print_arm(f"hybrid_r{round_i}", hybrid_metrics, hybrid_params)

            if mlp_metrics is not None and hybrid_metrics is not None:
                print(
                    format_eval_delta(
                        hybrid_metrics,
                        mlp_metrics,
                        candidate_name=f"hybrid_r{round_i}",
                        baseline_name=f"mlp_r{round_i}",
                    )
                )
                d_h = hybrid_metrics.top1_sf_disagree
                d_m = mlp_metrics.top1_sf_disagree
                if d_h is not None and d_m is not None:
                    print(
                        f"primary delta disagree_t1 (hybrid - mlp) round={round_i} = "
                        f"{float(d_h) - float(d_m):+.4f}"
                    )

            marked = registry.mark_processed(game_ids)
            if marked <= 0:
                logger.error(
                    "Fit succeeded but mark_processed updated 0 rows (games=%s)",
                    len(game_ids),
                )
                return 2

            n_after = store.count_unprocessed_train(split_version=split_version)
            elapsed_s = time.monotonic() - t0
            curve.append((round_i, last_id, len(datums), mlp_metrics, hybrid_metrics))
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
                    "mlp": mlp_metrics.as_dict() if mlp_metrics else None,
                    "hybrid": hybrid_metrics.as_dict() if hybrid_metrics else None,
                },
            )
            if n_after >= n_before:
                logger.error(
                    "Unprocessed count did not drop (before=%s after=%s)",
                    n_before,
                    n_after,
                )
                return 2

    print("\n=== encoder queue A/B curve ===")
    for r_i, last_id, n_train, m_m, h_m in curve:
        print(f"round={r_i} last_game_id={last_id} train_n={n_train}")
        if m_m is not None:
            print(format_eval_metrics(f"  mlp_r{r_i}", m_m))
        if h_m is not None:
            print(format_eval_metrics(f"  hybrid_r{r_i}", h_m))
    if not curve:
        return 1
    return 0 if len(curve) >= max_rounds else 3


def run_encoder_ab(
    *,
    limit: int,
    epochs: int,
    split_version: str,
    style_disagree_boost: float,
    style_disagree_scale: float,
    arms: str = "both",
    full_val: bool = False,
    max_rounds: int = 1,
    min_new_moves: int = MIN_NEW_MOVES_BASELINE,
    batch_limit: int | None = None,
    output_dir: str | Path | None = None,
    reset_queue: bool = False,
    parent_hybrid: str | Path | None = None,
    parent_mlp: str | Path | None = None,
) -> int:
    if max_rounds <= 1 and not reset_queue:
        return run_encoder_ab_single(
            limit=limit,
            epochs=epochs,
            split_version=split_version,
            style_disagree_boost=style_disagree_boost,
            style_disagree_scale=style_disagree_scale,
            arms=arms,
            full_val=full_val,
        )
    hyb = Path(parent_hybrid) if parent_hybrid else None
    mlp = Path(parent_mlp) if parent_mlp else None
    if hyb is not None and not hyb.is_file():
        logger.error("Missing --parent-hybrid %s", hyb)
        return 1
    if mlp is not None and not mlp.is_file():
        logger.error("Missing --parent-mlp %s", mlp)
        return 1
    return run_encoder_queue_catch_up(
        split_version=split_version,
        full_val=full_val,
        val_limit=limit,
        max_rounds=max_rounds,
        min_new_moves=min_new_moves,
        batch_limit=batch_limit or MAX_MOVES_PER_BASELINE_BATCH,
        epochs=epochs,
        style_disagree_boost=style_disagree_boost,
        style_disagree_scale=style_disagree_scale,
        arms=arms,
        output_dir=output_dir,
        reset_queue=reset_queue,
        parent_hybrid=hyb,
        parent_mlp=mlp,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=BaselineTrainer.DEFAULT_EPOCHS)
    parser.add_argument(
        "--split-version",
        "--salt",
        dest="split_version",
        type=str,
        default=DEFAULT_SPLIT_SALT,
    )
    parser.add_argument("--style-disagree-boost", type=float, default=2.0)
    parser.add_argument("--style-disagree-scale", type=float, default=2.0)
    parser.add_argument(
        "--arms",
        type=str,
        choices=("both", "mlp", "hybrid"),
        default="both",
    )
    parser.add_argument("--full-val", action="store_true")
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=1,
        help="1 = single sample A/B; >1 = registry-queue catch-up.",
    )
    parser.add_argument("--min-new-moves", type=int, default=MIN_NEW_MOVES_BASELINE)
    parser.add_argument("--batch-limit", type=int, default=MAX_MOVES_PER_BASELINE_BATCH)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--reset-queue",
        action="store_true",
        help="Clear already_processed_baseline before run (fresh game_id walk).",
    )
    parser.add_argument(
        "--parent-hybrid",
        type=str,
        default=None,
        help="Optional hybrid.keras to resume (cold-start if omitted).",
    )
    parser.add_argument(
        "--parent-mlp",
        type=str,
        default=None,
        help="Optional mlp.keras to resume (cold-start if omitted).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return run_encoder_ab(
        limit=max(50, args.limit),
        epochs=max(1, args.epochs),
        split_version=args.split_version,
        style_disagree_boost=args.style_disagree_boost,
        style_disagree_scale=args.style_disagree_scale,
        arms=args.arms,
        full_val=bool(args.full_val),
        max_rounds=max(1, int(args.max_rounds)),
        min_new_moves=max(1, int(args.min_new_moves)),
        batch_limit=max(1, int(args.batch_limit)),
        output_dir=str(args.output_dir) if args.output_dir else None,
        reset_queue=bool(args.reset_queue),
        parent_hybrid=str(args.parent_hybrid) if args.parent_hybrid else None,
        parent_mlp=str(args.parent_mlp) if args.parent_mlp else None,
    )
