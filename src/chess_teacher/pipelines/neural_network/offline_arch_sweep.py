"""Offline architecture sweep: cold-start each (hidden, score_hidden) cell.

Same frozen registry split for every cell. Primary metric: ``top1_sf_disagree``.
Logs ``CANDIDATE_MOVE_FEAT_VERSION``, ``MOVE_FEAT_DIM``, and Keras param count.
Feat layout is frozen - no feat-version CLI flags.
"""

from __future__ import annotations

import argparse
from typing import Any

from chess_teacher.pipelines.neural_network.candidate_eval import (
    CANDIDATE_MOVE_FEAT_VERSION,
    MOVE_FEAT_DIM,
)
from chess_teacher.pipelines.neural_network.eval_metrics import (
    EvalMetrics,
    evaluate_datums,
    format_eval_metrics,
)
from chess_teacher.pipelines.neural_network.offline_eval import load_registry_split
from chess_teacher.pipelines.neural_network.splits import DEFAULT_SPLIT_SALT, format_split_summary
from chess_teacher.pipelines.neural_network.train import BaselineTrainer
from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.logging import get_logger

logger = get_logger()

# (hidden, score_hidden) - first 2b capacity grid.
ARCH_SWEEP_GRID: tuple[tuple[int, int], ...] = (
    (128, 64),
    (128, 128),
    (256, 64),
    (256, 128),
)


def _keras_param_count(model: Any) -> int:
    fn = getattr(model, "count_params", None)
    if callable(fn):
        return int(fn())
    return 0


def run_arch_sweep(
    *,
    limit: int,
    epochs: int,
    split_version: str,
    style_disagree_boost: float,
    style_disagree_scale: float,
) -> int:
    db = get_db_client()
    logger.info(
        "Loading registry split once limit=%s split_version=%s feat_version=%s feat_dim=%s...",
        limit,
        split_version,
        CANDIDATE_MOVE_FEAT_VERSION,
        MOVE_FEAT_DIM,
    )
    split = load_registry_split(
        db,
        limit=limit,
        split_version=split_version,
        assign_if_missing=False,
    )
    print("\n" + format_split_summary(split))
    train = split.train_datums
    val = split.val_datums
    if len(train) < 30:
        logger.error("Train split too small: %s moves", len(train))
        return 1
    if len(val) < 10:
        logger.error("Val split too small: %s moves", len(val))
        return 1

    rows: list[tuple[int, int, int, EvalMetrics]] = []
    for hidden, score_hidden in ARCH_SWEEP_GRID:
        trainer = BaselineTrainer(
            epochs=epochs,
            hidden=hidden,
            score_hidden=score_hidden,
            style_disagree_boost=style_disagree_boost,
            style_disagree_scale=style_disagree_scale,
        )
        logger.info(
            "Cold-start arch hidden=%s score_hidden=%s epochs=%s feat_version=%s",
            hidden,
            score_hidden,
            epochs,
            CANDIDATE_MOVE_FEAT_VERSION,
        )
        model, _train_metrics = trainer.fit(train, weights_path=None)
        metrics = evaluate_datums(model, val)
        n_params = _keras_param_count(model)
        rows.append((hidden, score_hidden, n_params, metrics))
        logger.info(
            "arch hidden=%s score_hidden=%s params=%s feat_version=%s feat_dim=%s disagree_t1=%s",
            hidden,
            score_hidden,
            n_params,
            CANDIDATE_MOVE_FEAT_VERSION,
            MOVE_FEAT_DIM,
            metrics.top1_sf_disagree,
        )
        print(
            format_eval_metrics(f"{hidden}/{score_hidden}", metrics)
            + f" params={n_params} feat_version={CANDIDATE_MOVE_FEAT_VERSION} "
            f"feat_dim={MOVE_FEAT_DIM}"
        )

    def _disagree_key(row: tuple[int, int, int, EvalMetrics]) -> float:
        value = row[3].top1_sf_disagree
        return float("-inf") if value is None else float(value)

    ranked = sorted(rows, key=_disagree_key, reverse=True)
    print(
        "\n=== arch sweep (registry val, "
        f"feat_version={CANDIDATE_MOVE_FEAT_VERSION} feat_dim={MOVE_FEAT_DIM}) ==="
    )
    print(f"{'hid':>5} {'sc':>5} {'params':>8} {'top1':>7} {'dis_t1':>7} {'agree_t1':>8}")
    for hidden, score_hidden, n_params, metrics in ranked:
        dis = f"{metrics.top1_sf_disagree:.4f}" if metrics.top1_sf_disagree is not None else "n/a"
        agr = f"{metrics.top1_sf_agree:.4f}" if metrics.top1_sf_agree is not None else "n/a"
        print(
            f"{hidden:5d} {score_hidden:5d} {n_params:8d} "
            f"{metrics.top1_overall:7.4f} {dis:>7} {agr:>8}"
        )
    winner = ranked[0]
    print(
        f"\nPrimary winner (top1_sf_disagree): hidden={winner[0]} "
        f"score_hidden={winner[1]} params={winner[2]} "
        f"feat_version={CANDIDATE_MOVE_FEAT_VERSION} feat_dim={MOVE_FEAT_DIM}"
    )
    print("Feat layout frozen this sweep (no feat-version flags).")
    return 0


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
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    return run_arch_sweep(
        limit=max(50, int(args.limit)),
        epochs=max(1, int(args.epochs)),
        split_version=str(args.split_version),
        style_disagree_boost=float(args.style_disagree_boost),
        style_disagree_scale=float(args.style_disagree_scale),
    )


if __name__ == "__main__":
    from chess_teacher.utils.process_utils import run_script_main

    run_script_main(main)
