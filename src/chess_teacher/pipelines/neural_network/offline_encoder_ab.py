"""Offline encoder A/B: flat-state MLP vs hybrid board conv (Phase 2c / E20).

Same train sample, epochs, batch, and style weights for both arms.
One change family: representation only (no forced-weight / loss tweaks).

Primary metrics: stratified top1/top3 (overall / agree / disagree).
Prefer ``--full-val`` for decision runs (frozen registry val).
"""

from __future__ import annotations

import argparse
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
from chess_teacher.pipelines.neural_network.eval_metrics import (
    EvalMetrics,
    evaluate_datums,
    format_eval_delta,
    format_eval_metrics,
)
from chess_teacher.pipelines.neural_network.offline_eval import (
    load_registry_split,
    load_registry_val_datums,
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


def run_encoder_ab(
    *,
    limit: int,
    epochs: int,
    split_version: str,
    style_disagree_boost: float,
    style_disagree_scale: float,
    arms: str = "both",
    full_val: bool = False,
) -> int:
    """Cold-start MLP and/or hybrid; score on sample val or full registry val."""
    db = get_db_client()
    logger.info(
        "Encoder A/B load registry split limit=%s split_version=%s "
        "feat_version=%s board_version=%s arms=%s full_val=%s...",
        limit,
        split_version,
        CANDIDATE_MOVE_FEAT_VERSION,
        BOARD_TENSOR_VERSION,
        arms,
        full_val,
    )
    split = load_registry_split(
        db,
        limit=limit,
        split_version=split_version,
        assign_if_missing=False,
    )
    print("\n" + format_split_summary(split))
    train = split.train_datums
    if full_val:
        val = load_registry_val_datums(
            db,
            split_version=split_version,
            full=True,
            assign_if_missing=False,
        )
        logger.info("Using full registry val n=%s (ignoring sample val slice)", len(val))
        print(f"\nfull registry val moves={len(val)}")
    else:
        val = split.val_datums
    if len(train) < 30:
        logger.error("Train split too small: %s moves", len(train))
        return 1
    if len(val) < 10:
        logger.error("Val set too small: %s moves", len(val))
        return 1

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
        logger.info("Cold-start MLP control epochs=%s train_n=%s", epochs, len(train))
        model, _ = trainer.fit(train, weights_path=None)
        mlp_params = _keras_param_count(model)
        logger.info("Evaluating MLP on val_n=%s...", len(val))
        mlp_metrics = evaluate_datums(model, val)
        _print_arm("mlp", mlp_metrics, mlp_params)

    if arms in ("both", "hybrid"):
        trainer_h = HybridBoardTrainer(
            epochs=epochs,
            style_disagree_boost=style_disagree_boost,
            style_disagree_scale=style_disagree_scale,
        )
        logger.info("Cold-start hybrid board epochs=%s train_n=%s", epochs, len(train))
        model_h, _ = trainer_h.fit(train, weights_path=None)
        hybrid_params = _keras_param_count(model_h)
        logger.info("Evaluating hybrid on val_n=%s...", len(val))
        hybrid_metrics = evaluate_datums(model_h, val)
        _print_arm("hybrid", hybrid_metrics, hybrid_params)

    val_label = "full registry val" if full_val else "sample registry val"
    print(f"\n=== encoder A/B ({val_label}; change family = encoder only) ===")
    if mlp_metrics is not None and hybrid_metrics is not None:
        print(
            format_eval_delta(
                hybrid_metrics, mlp_metrics, candidate_name="hybrid", baseline_name="mlp"
            )
        )
        d_dis = hybrid_metrics.top1_sf_disagree
        m_dis = mlp_metrics.top1_sf_disagree
        if d_dis is not None and m_dis is not None:
            delta = float(d_dis) - float(m_dis)
            print(
                f"primary delta disagree_t1 (hybrid - mlp) = {delta:+.4f} "
                f"(hybrid_params={hybrid_params} mlp_params={mlp_params})"
            )
    print(
        "See .agents/docs/ml-phase2c-board-encoder.md for E21-E23 weight/loss/metric proposals. "
        "Forced-move weights default OFF for this A/B."
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Train sample move cap (oldest-N then train bucket). Val uses --full-val if set.",
    )
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
        help="Which cold-start arms to run (default both).",
    )
    parser.add_argument(
        "--full-val",
        action="store_true",
        help="Score all registry val games (official frozen val; ignores sample val slice).",
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
    )
