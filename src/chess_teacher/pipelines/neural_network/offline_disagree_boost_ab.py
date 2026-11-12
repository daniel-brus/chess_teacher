"""Offline A/B: platform style-disagree boost 2 vs 1 (Phase 5 Slice B).

Cold-start shallow hybrids on the same registry-train sample; only
``style_disagree_boost`` differs. Score both on the same registry val.
Does not write ``ml.baseline_models`` and does not read test.

Arms
----
``boost_2``
    Current default (control): disagree positions up-weighted x2 at full strength.
``boost_1``
    Flat style weight (no SF-disagree boost).

Gate vs ``boost_2``: clear ``val_log_loss`` drop or ~+1-2 top-1 / disagree lift;
pawn gap safety. Prefer promote on disagree + log-loss.

Smoke::

    .venv/Scripts/python.exe scripts/tools/offline_disagree_boost_ab.py \\
      --train-limit 2000 --val-limit 500 --epochs 2 \\
      --output storage/tmp/disagree_boost_ab_smoke.json

Mid decision::

    .venv/Scripts/python.exe scripts/tools/offline_disagree_boost_ab.py \\
      --train-limit 30000 --val-limit 8000 --epochs 3 \\
      --output storage/tmp/disagree_boost_ab_mid.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from chess_teacher.pipelines.neural_network.board_encoder import HybridBoardTrainer
from chess_teacher.pipelines.neural_network.create_training_set import TrainingDatum
from chess_teacher.pipelines.neural_network.hybrid_val_score import score_hybrid_datums
from chess_teacher.pipelines.neural_network.offline_eval import load_registry_bucket_datums
from chess_teacher.pipelines.neural_network.ply_weights import (
    DEFAULT_STYLE_DISAGREE_SCALE,
)
from chess_teacher.pipelines.neural_network.splits import SplitBucket
from chess_teacher.pipelines.neural_network.tf_runtime import ensure_tensorflow_logging
from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.logging import get_logger

logger = get_logger()

_ARMS = ("boost_2", "boost_1")
_ARM_BOOST = {"boost_2": 2.0, "boost_1": 1.0}


def _arm_payload(
    metrics: Any,
    log_loss: float,
    *,
    param_count: int,
    fit_seconds: float,
    n_train: int,
    train_loss_last: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "val": metrics.as_dict(),
        "val_log_loss": log_loss,
        "param_count": param_count,
        "fit_seconds": fit_seconds,
        "n_train": n_train,
        "train_loss_last": train_loss_last,
    }
    if extra:
        payload.update(extra)
    return payload


def _fit_arm(
    *,
    name: str,
    train: list[TrainingDatum],
    val: list[TrainingDatum],
    epochs: int,
    batch_size: int,
    style_disagree_boost: float,
    style_disagree_scale: float,
) -> dict[str, Any]:
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
    payload = _arm_payload(
        val_metrics,
        log_loss,
        param_count=int(model.count_params()),
        fit_seconds=elapsed,
        n_train=int(fit_metrics["n_samples"]),
        train_loss_last=float(fit_metrics.get("loss", float("nan"))),
        extra={
            "arm": name,
            "style_disagree_boost": float(style_disagree_boost),
            "style_disagree_scale": float(style_disagree_scale),
            "fit_epochs_completed": float(fit_metrics.get("epochs", epochs)),
        },
    )
    del model
    return payload


def run_disagree_boost_ab(
    *,
    train_limit: int | None,
    val_limit: int | None,
    epochs: int,
    arms: list[str],
    batch_size: int,
    style_disagree_scale: float,
    output: Path,
) -> dict[str, Any]:
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
        "Loading registry buckets train_limit=%s val_limit=%s (test unread)",
        train_limit,
        val_limit,
    )
    train = load_registry_bucket_datums(db, bucket=SplitBucket.TRAIN, limit=train_limit)
    val = load_registry_bucket_datums(db, bucket=SplitBucket.VAL, limit=val_limit)
    if len(train) < 8 or len(val) < 4:
        raise RuntimeError(f"buckets too small train={len(train)} val={len(val)}")

    report: dict[str, Any] = {
        "train_limit": train_limit,
        "val_limit": val_limit,
        "epochs": epochs,
        "batch_size": batch_size,
        "style_disagree_scale": style_disagree_scale,
        "n_train_loaded": len(train),
        "n_val_loaded": len(val),
        "cold_start": True,
        "test_read": False,
        "arms": {},
    }
    logger.info(
        "Disagree-boost A/B n_train=%s n_val=%s epochs=%s arms=%s",
        len(train),
        len(val),
        epochs,
        arms,
    )
    for name in arms:
        boost = float(_ARM_BOOST[name])
        logger.info("Running arm=%s style_disagree_boost=%s", name, boost)
        report["arms"][name] = _fit_arm(
            name=name,
            train=train,
            val=val,
            epochs=epochs,
            batch_size=batch_size,
            style_disagree_boost=boost,
            style_disagree_scale=style_disagree_scale,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote disagree-boost A/B report to %s", output)
    return report


def format_report(report: dict[str, Any]) -> str:
    lines = [
        (
            f"train={report['n_train_loaded']} val={report['n_val_loaded']} "
            f"epochs={report['epochs']} scale={report['style_disagree_scale']}"
        )
    ]
    control = (report.get("arms") or {}).get("boost_2")
    for name, arm in (report.get("arms") or {}).items():
        val = arm["val"]
        d_top1 = ""
        d_ll = ""
        d_dis = ""
        if control is not None and name != "boost_2":
            d_top1 = f" d_top1={val['top1_overall'] - control['val']['top1_overall']:+.4f}"
            d_ll = f" d_ll={arm['val_log_loss'] - control['val_log_loss']:+.4f}"
            c_dis = control["val"].get("top1_sf_disagree")
            v_dis = val.get("top1_sf_disagree")
            if c_dis is not None and v_dis is not None:
                d_dis = f" d_disagree={v_dis - c_dis:+.4f}"
        lines.append(
            f"{name}: boost={arm.get('style_disagree_boost', float('nan')):.1f} "
            f"top1={val['top1_overall']:.4f} "
            f"agree={val.get('top1_sf_agree', float('nan')):.4f} "
            f"disagree={val.get('top1_sf_disagree', float('nan')):.4f} "
            f"log_loss={arm['val_log_loss']:.4f} "
            f"pawn_gap={val['sf_delta_mean_pawns']:.4f} "
            f"fit_s={arm['fit_seconds']:.1f}{d_top1}{d_dis}{d_ll}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-limit", type=int, default=2000)
    parser.add_argument("--val-limit", type=int, default=1000)
    parser.add_argument(
        "--full-buckets",
        action="store_true",
        help="Load whole train/val buckets (still no test).",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--arms",
        default="boost_2,boost_1",
        help="Comma-separated: boost_2, boost_1",
    )
    parser.add_argument(
        "--style-disagree-scale",
        type=float,
        default=DEFAULT_STYLE_DISAGREE_SCALE,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("storage/tmp/disagree_boost_ab.json"),
    )
    args = parser.parse_args()
    arms = [part.strip() for part in str(args.arms).split(",") if part.strip()]
    train_limit = None if args.full_buckets else args.train_limit
    val_limit = None if args.full_buckets else args.val_limit
    report = run_disagree_boost_ab(
        train_limit=train_limit,
        val_limit=val_limit,
        epochs=args.epochs,
        arms=arms,
        batch_size=args.batch_size,
        style_disagree_scale=args.style_disagree_scale,
        output=args.output,
    )
    print("\n" + format_report(report))
    return 0
