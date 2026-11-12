"""Offline ABCD for hybrid warm-start update-rule (Phase 5 Slice A).

Shared cold parent on half of registry-train; four warm finetunes on the other
half; score all on the same registry val. Does not write ``ml.baseline_models``
and does not read test.

Arms
----
``control``
    Today: Adam 1e-3, last epoch, new batch only.
``lr_only``
    Adam 1e-4, fresh optimizer, last epoch, new batch only.
``lr_best_val``
    Adam 1e-4 + warm warmup + keep best registry-val epoch, new batch only.
``lr_best_val_replay``
    Same as ``lr_best_val``, plus ~1:1 mix of parent-batch + finetune-batch.

Smoke::

    .venv/Scripts/python.exe scripts/tools/offline_update_rule_ab.py \\
      --train-limit 2000 --val-limit 500 --epochs 2 \\
      --output storage/tmp/update_rule_ab_smoke.json

Mid decision::

    .venv/Scripts/python.exe scripts/tools/offline_update_rule_ab.py \\
      --train-limit 30000 --val-limit 8000 --epochs 3 \\
      --output storage/tmp/update_rule_ab_mid.json
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
import time
from pathlib import Path
from typing import Any

from chess_teacher.pipelines.neural_network.board_encoder import HybridBoardTrainer
from chess_teacher.pipelines.neural_network.create_training_set import TrainingDatum
from chess_teacher.pipelines.neural_network.hybrid_val_score import score_hybrid_datums
from chess_teacher.pipelines.neural_network.offline_eval import load_registry_bucket_datums
from chess_teacher.pipelines.neural_network.ply_weights import (
    DEFAULT_STYLE_DISAGREE_BOOST,
    DEFAULT_STYLE_DISAGREE_SCALE,
)
from chess_teacher.pipelines.neural_network.splits import SplitBucket
from chess_teacher.pipelines.neural_network.tf_runtime import ensure_tensorflow_logging
from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.logging import get_logger

logger = get_logger()

_ARMS = ("control", "lr_only", "lr_best_val", "lr_best_val_replay")


def split_train_by_game(
    datums: list[TrainingDatum],
    *,
    parent_frac: float = 0.5,
    seed: int = 0,
) -> tuple[list[TrainingDatum], list[TrainingDatum]]:
    """Split moves by ``game_id`` so games never cross parent/finetune."""
    if not 0.0 < float(parent_frac) < 1.0:
        raise ValueError(f"parent_frac must be in (0,1), got {parent_frac}")
    by_game: dict[str, list[TrainingDatum]] = {}
    for datum in datums:
        by_game.setdefault(str(datum.game_id), []).append(datum)
    game_ids = sorted(by_game.keys())
    rng = random.Random(int(seed))
    rng.shuffle(game_ids)
    n_parent_games = max(1, round(len(game_ids) * float(parent_frac)))
    n_parent_games = min(n_parent_games, len(game_ids) - 1) if len(game_ids) > 1 else 1
    parent_ids = set(game_ids[:n_parent_games])
    parent: list[TrainingDatum] = []
    finetune: list[TrainingDatum] = []
    for gid, rows in by_game.items():
        if gid in parent_ids:
            parent.extend(rows)
        else:
            finetune.extend(rows)
    if len(parent) < 8 or len(finetune) < 8:
        raise RuntimeError(
            f"split too small parent={len(parent)} finetune={len(finetune)} games={len(game_ids)}"
        )
    return parent, finetune


def mix_replay(
    old: list[TrainingDatum],
    new: list[TrainingDatum],
    *,
    seed: int = 0,
) -> list[TrainingDatum]:
    """~1:1 mix: take ``len(new)`` old rows (or all if fewer) + all new, shuffle."""
    rng = random.Random(int(seed))
    n_old = min(len(old), len(new))
    old_sample = list(old)
    rng.shuffle(old_sample)
    mixed = old_sample[:n_old] + list(new)
    rng.shuffle(mixed)
    return mixed


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
    parent_path: Path,
    epochs: int,
    batch_size: int,
    style_disagree_boost: float,
    style_disagree_scale: float,
    learning_rate: float,
    warm_learning_rate: float,
    warm_warmup_epochs: int,
    select_best_val: bool,
) -> dict[str, Any]:
    trainer = HybridBoardTrainer(
        epochs=epochs,
        batch_size=batch_size,
        style_disagree_boost=style_disagree_boost,
        style_disagree_scale=style_disagree_scale,
        learning_rate=learning_rate,
        warm_learning_rate=warm_learning_rate,
        warm_warmup_epochs=warm_warmup_epochs,
        select_best_val=select_best_val,
    )
    started = time.monotonic()
    model, fit_metrics = trainer.fit(
        train,
        weights_path=parent_path,
        require_parent_weights=True,
        val_datums=val if select_best_val else None,
    )
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
            "learning_rate_peak": float(fit_metrics.get("learning_rate", warm_learning_rate)),
            "warm_warmup_epochs": int(fit_metrics.get("warm_warmup_epochs", 0)),
            "select_best_val": bool(select_best_val),
            "best_val_epoch": fit_metrics.get("best_val_epoch"),
            "best_val_log_loss": fit_metrics.get("best_val_log_loss"),
            "fit_epochs_completed": float(fit_metrics.get("epochs", epochs)),
        },
    )
    del model
    return payload


def run_update_rule_ab(
    *,
    train_limit: int | None,
    val_limit: int | None,
    epochs: int,
    arms: list[str],
    batch_size: int,
    style_disagree_boost: float,
    style_disagree_scale: float,
    output: Path,
    parent_frac: float = 0.5,
    seed: int = 0,
    parent_epochs: int | None = None,
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
    train_all = load_registry_bucket_datums(db, bucket=SplitBucket.TRAIN, limit=train_limit)
    val = load_registry_bucket_datums(db, bucket=SplitBucket.VAL, limit=val_limit)
    parent_batch, finetune_batch = split_train_by_game(
        train_all,
        parent_frac=parent_frac,
        seed=seed,
    )
    cold_epochs = int(parent_epochs) if parent_epochs is not None else int(epochs)
    logger.info(
        "ABCD split parent=%s finetune=%s val=%s cold_epochs=%s warm_epochs=%s arms=%s",
        len(parent_batch),
        len(finetune_batch),
        len(val),
        cold_epochs,
        epochs,
        arms,
    )

    report: dict[str, Any] = {
        "train_limit": train_limit,
        "val_limit": val_limit,
        "epochs_warm": epochs,
        "epochs_parent_cold": cold_epochs,
        "batch_size": batch_size,
        "style_disagree_boost": style_disagree_boost,
        "style_disagree_scale": style_disagree_scale,
        "parent_frac": parent_frac,
        "seed": seed,
        "n_train_loaded": len(train_all),
        "n_parent": len(parent_batch),
        "n_finetune": len(finetune_batch),
        "n_val_loaded": len(val),
        "cold_start": False,
        "test_read": False,
        "arms": {},
    }

    with tempfile.TemporaryDirectory(prefix="ct-update-rule-ab-") as tmp:
        parent_path = Path(tmp) / "parent.keras"
        parent_trainer = HybridBoardTrainer(
            epochs=cold_epochs,
            batch_size=batch_size,
            style_disagree_boost=style_disagree_boost,
            style_disagree_scale=style_disagree_scale,
            learning_rate=HybridBoardTrainer.DEFAULT_LEARNING_RATE,
        )
        parent_started = time.monotonic()
        parent_model, parent_metrics = parent_trainer.fit(parent_batch)
        HybridBoardTrainer.save(parent_model, parent_path)
        report["parent"] = {
            "fit_seconds": time.monotonic() - parent_started,
            "n_train": int(parent_metrics["n_samples"]),
            "train_loss_last": float(parent_metrics.get("loss", float("nan"))),
            "param_count": int(parent_model.count_params()),
        }
        # Parent alone on val (reference before any warm update).
        parent_val, parent_ll = score_hybrid_datums(parent_model, val)
        report["parent"]["val"] = parent_val.as_dict()
        report["parent"]["val_log_loss"] = parent_ll
        del parent_model

        arm_specs: dict[str, dict[str, Any]] = {
            "control": {
                "learning_rate": HybridBoardTrainer.DEFAULT_LEARNING_RATE,
                "warm_learning_rate": HybridBoardTrainer.DEFAULT_LEARNING_RATE,
                "warm_warmup_epochs": 0,
                "select_best_val": False,
                "replay": False,
            },
            "lr_only": {
                "learning_rate": HybridBoardTrainer.DEFAULT_LEARNING_RATE,
                "warm_learning_rate": HybridBoardTrainer.DEFAULT_WARM_LEARNING_RATE,
                "warm_warmup_epochs": 0,
                "select_best_val": False,
                "replay": False,
            },
            "lr_best_val": {
                "learning_rate": HybridBoardTrainer.DEFAULT_LEARNING_RATE,
                "warm_learning_rate": HybridBoardTrainer.DEFAULT_WARM_LEARNING_RATE,
                "warm_warmup_epochs": HybridBoardTrainer.DEFAULT_WARM_WARMUP_EPOCHS,
                "select_best_val": True,
                "replay": False,
            },
            "lr_best_val_replay": {
                "learning_rate": HybridBoardTrainer.DEFAULT_LEARNING_RATE,
                "warm_learning_rate": HybridBoardTrainer.DEFAULT_WARM_LEARNING_RATE,
                "warm_warmup_epochs": HybridBoardTrainer.DEFAULT_WARM_WARMUP_EPOCHS,
                "select_best_val": True,
                "replay": True,
            },
        }

        for name in arms:
            spec = arm_specs[name]
            train = (
                mix_replay(parent_batch, finetune_batch, seed=seed + 1)
                if spec["replay"]
                else finetune_batch
            )
            logger.info(
                "Running arm=%s n_train=%s warm_lr=%s warmup=%s best_val=%s replay=%s",
                name,
                len(train),
                spec["warm_learning_rate"],
                spec["warm_warmup_epochs"],
                spec["select_best_val"],
                spec["replay"],
            )
            report["arms"][name] = _fit_arm(
                name=name,
                train=train,
                val=val,
                parent_path=parent_path,
                epochs=epochs,
                batch_size=batch_size,
                style_disagree_boost=style_disagree_boost,
                style_disagree_scale=style_disagree_scale,
                learning_rate=float(spec["learning_rate"]),
                warm_learning_rate=float(spec["warm_learning_rate"]),
                warm_warmup_epochs=int(spec["warm_warmup_epochs"]),
                select_best_val=bool(spec["select_best_val"]),
            )
            report["arms"][name]["replay"] = bool(spec["replay"])
            report["arms"][name]["n_train_datums"] = len(train)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote update-rule A/B report to %s", output)
    return report


def format_report(report: dict[str, Any]) -> str:
    lines = [
        (
            f"parent={report['n_parent']} finetune={report['n_finetune']} "
            f"val={report['n_val_loaded']} warm_epochs={report['epochs_warm']} "
            f"parent_cold_epochs={report['epochs_parent_cold']}"
        )
    ]
    parent = report.get("parent") or {}
    if parent.get("val"):
        pv = parent["val"]
        lines.append(
            f"parent_cold: top1={pv['top1_overall']:.4f} "
            f"disagree={pv.get('top1_sf_disagree', float('nan')):.4f} "
            f"log_loss={parent.get('val_log_loss', float('nan')):.4f}"
        )
    control = (report.get("arms") or {}).get("control")
    for name, arm in (report.get("arms") or {}).items():
        val = arm["val"]
        d_top1 = ""
        d_ll = ""
        if control is not None and name != "control":
            d_top1 = f" d_top1={val['top1_overall'] - control['val']['top1_overall']:+.4f}"
            d_ll = f" d_ll={arm['val_log_loss'] - control['val_log_loss']:+.4f}"
        lines.append(
            f"{name}: top1={val['top1_overall']:.4f} "
            f"agree={val.get('top1_sf_agree', float('nan')):.4f} "
            f"disagree={val.get('top1_sf_disagree', float('nan')):.4f} "
            f"log_loss={arm['val_log_loss']:.4f} "
            f"pawn_gap={val['sf_delta_mean_pawns']:.4f} "
            f"n_train={arm.get('n_train_datums', arm['n_train'])} "
            f"fit_s={arm['fit_seconds']:.1f}{d_top1}{d_ll}"
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
    parser.add_argument("--epochs", type=int, default=3, help="Warm-start epochs per arm.")
    parser.add_argument(
        "--parent-epochs",
        type=int,
        default=None,
        help="Cold parent epochs (default: same as --epochs).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--arms",
        default="control,lr_only,lr_best_val,lr_best_val_replay",
        help="Comma-separated arm names.",
    )
    parser.add_argument("--parent-frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--style-disagree-boost", type=float, default=DEFAULT_STYLE_DISAGREE_BOOST)
    parser.add_argument("--style-disagree-scale", type=float, default=DEFAULT_STYLE_DISAGREE_SCALE)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("storage/tmp/update_rule_ab.json"),
    )
    args = parser.parse_args()
    arms = [part.strip() for part in str(args.arms).split(",") if part.strip()]
    train_limit = None if args.full_buckets else args.train_limit
    val_limit = None if args.full_buckets else args.val_limit
    report = run_update_rule_ab(
        train_limit=train_limit,
        val_limit=val_limit,
        epochs=args.epochs,
        arms=arms,
        batch_size=args.batch_size,
        style_disagree_boost=args.style_disagree_boost,
        style_disagree_scale=args.style_disagree_scale,
        output=args.output,
        parent_frac=args.parent_frac,
        seed=args.seed,
        parent_epochs=args.parent_epochs,
    )
    print("\n" + format_report(report))
    return 0
