"""Offline A/B: personal head full FT vs freeze board trunk (Phase 5 Slice C).

Shared cold parent on platform registry-train. Both arms warm-start from that
parent on one account ∩ registry-train; only ``freeze_board_trunk`` differs.
Score both on that account ∩ registry-val. No ``ml.baseline_models`` write,
no test bucket.

Arms
----
``full_ft``
    Current product path (control): all layers trainable.
``freeze_trunk``
    Freeze ``board_conv*`` / ``board_gap`` / ``board_emb``; train state + head.

Gate vs ``full_ft``: clear ``val_log_loss`` drop or ~+1-2 top-1 / disagree on
user val; pawn gap safety.

Smoke::

    .venv/Scripts/python.exe scripts/tools/offline_personal_head_ab.py \\
      --parent-train-limit 2000 --user-train-limit 1500 --val-limit 500 \\
      --parent-epochs 2 --epochs 2 \\
      --output storage/tmp/personal_head_ab_smoke.json

Mid decision::

    .venv/Scripts/python.exe scripts/tools/offline_personal_head_ab.py \\
      --parent-train-limit 30000 --user-train-limit 15000 --val-limit 4000 \\
      --parent-epochs 3 --epochs 3 \\
      --output storage/tmp/personal_head_ab_mid.json
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from chess_teacher.pipelines.neural_network.board_encoder import HybridBoardTrainer
from chess_teacher.pipelines.neural_network.create_training_set import TrainingDatum
from chess_teacher.pipelines.neural_network.hybrid_val_score import score_hybrid_datums
from chess_teacher.pipelines.neural_network.offline_eval import (
    load_account_registry_bucket_datums,
    load_registry_bucket_datums,
)
from chess_teacher.pipelines.neural_network.offline_user_finetune import (
    DEFAULT_USER_STYLE_DISAGREE_BOOST,
    MIN_USER_TRAIN_MOVES,
)
from chess_teacher.pipelines.neural_network.ply_weights import (
    DEFAULT_STYLE_DISAGREE_BOOST,
    DEFAULT_STYLE_DISAGREE_SCALE,
)
from chess_teacher.pipelines.neural_network.splits import DEFAULT_SPLIT_SALT, SplitBucket
from chess_teacher.pipelines.neural_network.tf_runtime import ensure_tensorflow_logging
from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.logging import get_logger

logger = get_logger()

# ikbendaniel — default personalization subject for Slice C mid runs.
_DEFAULT_ACCOUNT_ID = "5c3f069730e098aedc61b978c92d6448eebfdf234e575552e86343a10bfb698b"
_ARMS = ("full_ft", "freeze_trunk")
_ARM_FREEZE = {"full_ft": False, "freeze_trunk": True}


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
    freeze_board_trunk: bool,
) -> dict[str, Any]:
    # Match Slice A control / product warm path: Adam 1e-3, no soft warmup.
    # Only freeze_board_trunk should differ across arms.
    trainer = HybridBoardTrainer(
        epochs=epochs,
        batch_size=batch_size,
        style_disagree_boost=style_disagree_boost,
        style_disagree_scale=style_disagree_scale,
        learning_rate=HybridBoardTrainer.DEFAULT_LEARNING_RATE,
        warm_learning_rate=HybridBoardTrainer.DEFAULT_LEARNING_RATE,
        warm_warmup_epochs=0,
        freeze_board_trunk=freeze_board_trunk,
    )
    started = time.monotonic()
    model, fit_metrics = trainer.fit(
        train,
        weights_path=parent_path,
        require_parent_weights=True,
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
            "freeze_board_trunk": bool(freeze_board_trunk),
            "style_disagree_boost": float(style_disagree_boost),
            "style_disagree_scale": float(style_disagree_scale),
            "fit_epochs_completed": float(fit_metrics.get("epochs", epochs)),
            "freeze_board_trunk_metric": float(fit_metrics.get("freeze_board_trunk", 0.0)),
        },
    )
    del model
    return payload


def run_personal_head_ab(
    *,
    account_id: str,
    parent_train_limit: int | None,
    user_train_limit: int | None,
    val_limit: int | None,
    parent_epochs: int,
    epochs: int,
    arms: list[str],
    batch_size: int,
    parent_style_disagree_boost: float,
    user_style_disagree_boost: float,
    style_disagree_scale: float,
    split_version: str,
    output: Path,
    min_train_moves: int = MIN_USER_TRAIN_MOVES,
) -> dict[str, Any]:
    unknown = [name for name in arms if name not in _ARMS]
    if unknown:
        raise ValueError(f"Unknown arms {unknown}; choose from {_ARMS}")
    if not arms:
        raise ValueError("At least one arm is required")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if parent_epochs < 1:
        raise ValueError(f"parent_epochs must be >= 1, got {parent_epochs}")
    aid = (account_id or "").strip()
    if not aid:
        raise ValueError("account_id is required")

    ensure_tensorflow_logging()
    db = get_db_client()
    logger.info(
        "Loading platform parent train_limit=%s; user account=%s user_train=%s val=%s",
        parent_train_limit,
        aid[:12],
        user_train_limit,
        val_limit,
    )
    parent_batch = load_registry_bucket_datums(
        db, bucket=SplitBucket.TRAIN, limit=parent_train_limit, split_version=split_version
    )
    user_train = load_account_registry_bucket_datums(
        aid,
        db,
        bucket=SplitBucket.TRAIN,
        split_version=split_version,
        limit=user_train_limit,
    )
    user_val = load_account_registry_bucket_datums(
        aid,
        db,
        bucket=SplitBucket.VAL,
        split_version=split_version,
        limit=val_limit,
    )
    if len(parent_batch) < 8:
        raise RuntimeError(f"parent train too small: {len(parent_batch)}")
    if len(user_train) < min_train_moves:
        raise RuntimeError(
            f"user registry-train below min gate: n={len(user_train)} < {min_train_moves}"
        )
    if len(user_val) < 10:
        raise RuntimeError(f"user registry-val too small: {len(user_val)}")

    report: dict[str, Any] = {
        "account_id": aid,
        "split_version": split_version,
        "parent_train_limit": parent_train_limit,
        "user_train_limit": user_train_limit,
        "val_limit": val_limit,
        "parent_epochs": parent_epochs,
        "epochs_warm": epochs,
        "batch_size": batch_size,
        "parent_style_disagree_boost": parent_style_disagree_boost,
        "user_style_disagree_boost": user_style_disagree_boost,
        "style_disagree_scale": style_disagree_scale,
        "n_parent_train": len(parent_batch),
        "n_user_train": len(user_train),
        "n_user_val": len(user_val),
        "cold_start_parent": True,
        "test_read": False,
        "arms": {},
    }
    logger.info(
        "Personal-head A/B parent=%s user_train=%s user_val=%s parent_ep=%s warm_ep=%s arms=%s",
        len(parent_batch),
        len(user_train),
        len(user_val),
        parent_epochs,
        epochs,
        arms,
    )

    with tempfile.TemporaryDirectory(prefix="ct-personal-head-ab-") as tmp:
        parent_path = Path(tmp) / "parent.keras"
        parent_trainer = HybridBoardTrainer(
            epochs=parent_epochs,
            batch_size=batch_size,
            style_disagree_boost=parent_style_disagree_boost,
            style_disagree_scale=style_disagree_scale,
        )
        parent_started = time.monotonic()
        parent_model, parent_metrics = parent_trainer.fit(parent_batch)
        HybridBoardTrainer.save(parent_model, parent_path)
        parent_val, parent_ll = score_hybrid_datums(parent_model, user_val)
        report["parent"] = {
            "fit_seconds": time.monotonic() - parent_started,
            "n_train": int(parent_metrics["n_samples"]),
            "train_loss_last": float(parent_metrics.get("loss", float("nan"))),
            "param_count": int(parent_model.count_params()),
            "val": parent_val.as_dict(),
            "val_log_loss": parent_ll,
        }
        del parent_model

        for name in arms:
            freeze = bool(_ARM_FREEZE[name])
            logger.info(
                "Running arm=%s freeze_board_trunk=%s n_train=%s boost=%s",
                name,
                freeze,
                len(user_train),
                user_style_disagree_boost,
            )
            report["arms"][name] = _fit_arm(
                name=name,
                train=user_train,
                val=user_val,
                parent_path=parent_path,
                epochs=epochs,
                batch_size=batch_size,
                style_disagree_boost=user_style_disagree_boost,
                style_disagree_scale=style_disagree_scale,
                freeze_board_trunk=freeze,
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote personal-head A/B report to %s", output)
    return report


def format_report(report: dict[str, Any]) -> str:
    lines = [
        (
            f"account={str(report.get('account_id', ''))[:12]}… "
            f"parent={report['n_parent_train']} user_train={report['n_user_train']} "
            f"user_val={report['n_user_val']} warm_epochs={report['epochs_warm']} "
            f"user_boost={report['user_style_disagree_boost']}"
        )
    ]
    parent = report.get("parent") or {}
    if parent.get("val"):
        pv = parent["val"]
        lines.append(
            f"parent_on_user_val: top1={pv['top1_overall']:.4f} "
            f"disagree={pv.get('top1_sf_disagree', float('nan')):.4f} "
            f"log_loss={parent.get('val_log_loss', float('nan')):.4f} "
            f"pawn_gap={pv['sf_delta_mean_pawns']:.4f}"
        )
    control = (report.get("arms") or {}).get("full_ft")
    for name, arm in (report.get("arms") or {}).items():
        val = arm["val"]
        d_top1 = ""
        d_ll = ""
        d_dis = ""
        if control is not None and name != "full_ft":
            d_top1 = f" d_top1={val['top1_overall'] - control['val']['top1_overall']:+.4f}"
            d_ll = f" d_ll={arm['val_log_loss'] - control['val_log_loss']:+.4f}"
            c_dis = control["val"].get("top1_sf_disagree")
            v_dis = val.get("top1_sf_disagree")
            if c_dis is not None and v_dis is not None:
                d_dis = f" d_disagree={v_dis - c_dis:+.4f}"
        lines.append(
            f"{name}: freeze={arm.get('freeze_board_trunk')} "
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
    parser.add_argument("--account-id", type=str, default=_DEFAULT_ACCOUNT_ID)
    parser.add_argument("--split-version", type=str, default=DEFAULT_SPLIT_SALT)
    parser.add_argument("--parent-train-limit", type=int, default=2000)
    parser.add_argument("--user-train-limit", type=int, default=1500)
    parser.add_argument("--val-limit", type=int, default=500)
    parser.add_argument(
        "--full-buckets",
        action="store_true",
        help="Load whole parent/user train and user val (still no test).",
    )
    parser.add_argument("--parent-epochs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2, help="Warm finetune epochs per arm.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--arms",
        default="full_ft,freeze_trunk",
        help="Comma-separated: full_ft, freeze_trunk",
    )
    parser.add_argument(
        "--parent-style-disagree-boost",
        type=float,
        default=DEFAULT_STYLE_DISAGREE_BOOST,
    )
    parser.add_argument(
        "--user-style-disagree-boost",
        type=float,
        default=DEFAULT_USER_STYLE_DISAGREE_BOOST,
    )
    parser.add_argument(
        "--style-disagree-scale",
        type=float,
        default=DEFAULT_STYLE_DISAGREE_SCALE,
    )
    parser.add_argument(
        "--min-train-moves",
        type=int,
        default=MIN_USER_TRAIN_MOVES,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("storage/tmp/personal_head_ab.json"),
    )
    args = parser.parse_args()
    arms = [part.strip() for part in str(args.arms).split(",") if part.strip()]
    parent_train_limit = None if args.full_buckets else args.parent_train_limit
    user_train_limit = None if args.full_buckets else args.user_train_limit
    val_limit = None if args.full_buckets else args.val_limit
    report = run_personal_head_ab(
        account_id=str(args.account_id),
        parent_train_limit=parent_train_limit,
        user_train_limit=user_train_limit,
        val_limit=val_limit,
        parent_epochs=int(args.parent_epochs),
        epochs=int(args.epochs),
        arms=arms,
        batch_size=int(args.batch_size),
        parent_style_disagree_boost=float(args.parent_style_disagree_boost),
        user_style_disagree_boost=float(args.user_style_disagree_boost),
        style_disagree_scale=float(args.style_disagree_scale),
        split_version=str(args.split_version),
        output=args.output,
        min_train_moves=int(args.min_train_moves),
    )
    print("\n" + format_report(report))
    return 0
