"""Offline A/B: hybrid candidate scorer vs a small spatial policy.

Reads registry train and val only. Does not write ``ml.baseline_models``, does
not reset the training queue, and does not load the test bucket. Both arms are
cold starts on the same games, epochs, batch size, and style-disagree weights.
The reported scores are the last epoch on the val bucket.

Smoke (this cloud DB is enough)::

    .venv/bin/python scripts/tools/offline_maia_style_ab.py \\
      --train-limit 2000 --val-limit 500 --epochs 2 \\
      --output storage/tmp/maia_style_ab_smoke.json

Decision run on the machine that has the full split (still no test, no promote)::

    .venv/bin/python scripts/tools/offline_maia_style_ab.py --full-buckets --epochs 2 \\
      --output storage/tmp/maia_style_ab_full.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from chess_teacher.pipelines.neural_network.board_encoder import HybridBoardTrainer
from chess_teacher.pipelines.neural_network.eval_metrics import (
    EvalMetrics,
    compute_candidate_style_metrics,
    pack_datums_for_eval,
    predict_candidate_logits,
)
from chess_teacher.pipelines.neural_network.offline_eval import load_registry_bucket_datums
from chess_teacher.pipelines.neural_network.ply_weights import (
    DEFAULT_STYLE_DISAGREE_BOOST,
    DEFAULT_STYLE_DISAGREE_SCALE,
)
from chess_teacher.pipelines.neural_network.spatial_policy import (
    SpatialPolicyTrainer,
    masked_candidate_log_loss,
    score_spatial_datums,
)
from chess_teacher.pipelines.neural_network.splits import SplitBucket
from chess_teacher.pipelines.neural_network.tf_runtime import ensure_tensorflow_logging
from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.logging import get_logger

logger = get_logger()

_ARMS = ("hybrid", "spatial")


def score_hybrid_datums(model: Any, datums: list[Any]) -> tuple[EvalMetrics, float]:
    """Candidate-slot metrics and 128-way log-loss for a hybrid model."""
    packed = pack_datums_for_eval(datums)
    logits = predict_candidate_logits(model, packed.kept_datums, packed.feats)
    metrics = compute_candidate_style_metrics(
        logits=logits,
        mask=packed.mask,
        labels=packed.labels,
        move_feats=packed.feats,
        plies=[d.ply for d in packed.kept_datums],
        n_input=packed.n_input,
        max_candidates=packed.max_candidates,
    )
    log_loss = masked_candidate_log_loss(logits, packed.mask, packed.labels)
    return metrics, log_loss


def _arm_payload(
    metrics: EvalMetrics,
    log_loss: float,
    *,
    param_count: int,
    fit_seconds: float,
    n_train: int,
    train_loss_last: float,
    epoch_losses: list[float] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "val": metrics.as_dict(),
        "val_log_loss": log_loss,
        "param_count": param_count,
        "fit_seconds": fit_seconds,
        "n_train": n_train,
        "train_loss_last": train_loss_last,
    }
    if epoch_losses is not None:
        payload["epoch_losses"] = epoch_losses
    return payload


def run_maia_style_ab(
    *,
    train_limit: int | None,
    val_limit: int | None,
    epochs: int,
    arms: list[str],
    batch_size: int,
    style_disagree_boost: float,
    style_disagree_scale: float,
    output: Path,
) -> dict[str, Any]:
    """Train the requested arms and write the JSON report. Read-only on Postgres."""
    unknown = [name for name in arms if name not in _ARMS]
    if unknown:
        raise ValueError(f"Unknown arms {unknown}; choose from {_ARMS}")
    if not arms:
        raise ValueError("At least one arm is required")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")

    ensure_tensorflow_logging()
    db = get_db_client()
    logger.info(
        "Loading registry buckets train_limit=%s val_limit=%s (test is not read)",
        train_limit,
        val_limit,
    )
    train = load_registry_bucket_datums(db, bucket=SplitBucket.TRAIN, limit=train_limit)
    val = load_registry_bucket_datums(db, bucket=SplitBucket.VAL, limit=val_limit)
    if len(train) < 8:
        raise RuntimeError(f"Train bucket too small for a smoke: {len(train)} moves")
    if len(val) < 4:
        raise RuntimeError(f"Val bucket too small for a smoke: {len(val)} moves")

    report: dict[str, Any] = {
        "train_limit": train_limit,
        "val_limit": val_limit,
        "epochs": epochs,
        "batch_size": batch_size,
        "style_disagree_boost": style_disagree_boost,
        "style_disagree_scale": style_disagree_scale,
        "n_train_loaded": len(train),
        "n_val_loaded": len(val),
        "learning_rate": 1e-3,
        "cold_start": True,
        "test_read": False,
        "arms": {},
    }

    if "hybrid" in arms:
        trainer = HybridBoardTrainer(
            epochs=epochs,
            batch_size=batch_size,
            style_disagree_boost=style_disagree_boost,
            style_disagree_scale=style_disagree_scale,
        )
        started = time.monotonic()
        model, fit_metrics = trainer.fit(train)
        elapsed = time.monotonic() - started
        val_metrics, log_loss = score_hybrid_datums(model, val)
        report["arms"]["hybrid"] = _arm_payload(
            val_metrics,
            log_loss,
            param_count=int(model.count_params()),
            fit_seconds=elapsed,
            n_train=int(fit_metrics["n_samples"]),
            train_loss_last=float(fit_metrics.get("loss", float("nan"))),
        )
        del model

    if "spatial" in arms:
        trainer_s = SpatialPolicyTrainer(
            epochs=epochs,
            batch_size=batch_size,
            style_disagree_boost=style_disagree_boost,
            style_disagree_scale=style_disagree_scale,
        )
        model_s, fit_metrics_s = trainer_s.fit(train)
        val_metrics_s, log_loss_s, _logits, _mask, _labels = score_spatial_datums(
            model_s,
            val,
            batch_size=batch_size,
        )
        report["arms"]["spatial"] = _arm_payload(
            val_metrics_s,
            log_loss_s,
            param_count=int(fit_metrics_s["param_count"]),
            fit_seconds=float(fit_metrics_s["fit_seconds"]),
            n_train=int(fit_metrics_s["n_samples"]),
            train_loss_last=float(fit_metrics_s["loss"]),
            epoch_losses=list(fit_metrics_s["epoch_losses"]),
        )
        del model_s

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote Maia-style A/B report to %s", output)
    return report


def format_report(report: dict[str, Any]) -> str:
    """One text block for the terminal."""
    lines = [
        (
            f"train_loaded={report['n_train_loaded']} val_loaded={report['n_val_loaded']} "
            f"epochs={report['epochs']} lr={report['learning_rate']}"
        )
    ]
    for name, arm in report["arms"].items():
        val = arm["val"]
        lines.append(
            f"{name}: top1={val['top1_overall']:.4f} "
            f"agree={val.get('top1_sf_agree', float('nan')):.4f} "
            f"disagree={val.get('top1_sf_disagree', float('nan')):.4f} "
            f"log_loss={arm['val_log_loss']:.4f} "
            f"pawn_gap={val['sf_delta_mean_pawns']:.4f} "
            f"n_eval={int(val['n_eval'])} params={arm['param_count']} "
            f"fit_s={arm['fit_seconds']:.1f}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-limit", type=int, default=2000)
    parser.add_argument("--val-limit", type=int, default=1000)
    parser.add_argument(
        "--full-buckets",
        action="store_true",
        help="Load the whole train and val buckets. Still does not read test.",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--arms", default="hybrid,spatial")
    parser.add_argument("--style-disagree-boost", type=float, default=DEFAULT_STYLE_DISAGREE_BOOST)
    parser.add_argument("--style-disagree-scale", type=float, default=DEFAULT_STYLE_DISAGREE_SCALE)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("storage/tmp/maia_style_ab.json"),
    )
    args = parser.parse_args()
    arms = [part.strip() for part in str(args.arms).split(",") if part.strip()]
    train_limit = None if args.full_buckets else args.train_limit
    val_limit = None if args.full_buckets else args.val_limit
    report = run_maia_style_ab(
        train_limit=train_limit,
        val_limit=val_limit,
        epochs=args.epochs,
        arms=arms,
        batch_size=args.batch_size,
        style_disagree_boost=args.style_disagree_boost,
        style_disagree_scale=args.style_disagree_scale,
        output=args.output,
    )
    print("\n" + format_report(report))
    return 0
