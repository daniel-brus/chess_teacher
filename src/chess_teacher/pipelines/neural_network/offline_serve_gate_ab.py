"""Offline A/B: serve parent vs personal FT boosts (Phase 5 Slice D).

Shared cold parent on platform registry-train. Warm FT arms on one account ∩
registry-train at different style-disagree boosts; score all on that account ∩
registry-val. ``serve_parent`` is the no-FT reference. No ``ml.baseline_models``
write, no test bucket.

Arms
----
``serve_parent``
    Parent weights on user val (reference).
``ft_boost_1`` / ``ft_boost_2`` / ``ft_boost_4``
    Full-layer user FT at style disagree boost 1 / 2 / 4.

Draft serve gate vs parent (all must hold): disagree lift ≥ +1.0 pt, overall
top-1 drop ≤ 1.0 pt, pawn-gap regress ≤ 0.20, flag ΔLL > 0.03.

Smoke::

    .venv/Scripts/python.exe scripts/tools/offline_serve_gate_ab.py \\
      --parent-train-limit 2000 --user-train-limit 1500 --val-limit 500 \\
      --parent-epochs 2 --epochs 2 \\
      --output storage/tmp/serve_gate_ab_smoke.json

Mid decision::

    .venv/Scripts/python.exe scripts/tools/offline_serve_gate_ab.py \\
      --parent-train-limit 30000 --user-train-limit 15000 --val-limit 4000 \\
      --parent-epochs 3 --epochs 3 \\
      --output storage/tmp/serve_gate_ab_mid.json
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
    MIN_USER_TRAIN_MOVES,
)
from chess_teacher.pipelines.neural_network.ply_weights import (
    DEFAULT_STYLE_DISAGREE_BOOST,
    DEFAULT_STYLE_DISAGREE_SCALE,
)
from chess_teacher.pipelines.neural_network.serve_gate import (
    SERVE_GATE_LL_WARN,
    SERVE_GATE_MAX_OVERALL_DROP,
    SERVE_GATE_MAX_PAWN_GAP_REGRESS,
    SERVE_GATE_MIN_DISAGREE_LIFT,
    evaluate_serve_gate,
)
from chess_teacher.pipelines.neural_network.splits import DEFAULT_SPLIT_SALT, SplitBucket
from chess_teacher.pipelines.neural_network.tf_runtime import ensure_tensorflow_logging
from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.logging import get_logger

logger = get_logger()

# Re-export for older imports / tests.
__all__ = (
    "SERVE_GATE_LL_WARN",
    "SERVE_GATE_MAX_OVERALL_DROP",
    "SERVE_GATE_MAX_PAWN_GAP_REGRESS",
    "SERVE_GATE_MIN_DISAGREE_LIFT",
    "evaluate_serve_gate",
    "format_report",
    "main",
    "run_serve_gate_ab",
)

# ikbendaniel — default personalization subject (same as Slice C).
_DEFAULT_ACCOUNT_ID = "5c3f069730e098aedc61b978c92d6448eebfdf234e575552e86343a10bfb698b"
_ARMS = ("serve_parent", "ft_boost_1", "ft_boost_2", "ft_boost_4")
_ARM_BOOST: dict[str, float | None] = {
    "serve_parent": None,
    "ft_boost_1": 1.0,
    "ft_boost_2": 2.0,
    "ft_boost_4": 4.0,
}


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


def _fit_ft_arm(
    *,
    name: str,
    train: list[TrainingDatum],
    val: list[TrainingDatum],
    parent_path: Path,
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
        learning_rate=HybridBoardTrainer.DEFAULT_LEARNING_RATE,
        warm_learning_rate=HybridBoardTrainer.DEFAULT_LEARNING_RATE,
        warm_warmup_epochs=0,
        freeze_board_trunk=False,
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
            "style_disagree_boost": float(style_disagree_boost),
            "style_disagree_scale": float(style_disagree_scale),
            "fit_epochs_completed": float(fit_metrics.get("epochs", epochs)),
            "served_parent": False,
        },
    )
    del model
    return payload


def run_serve_gate_ab(
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
    if "serve_parent" not in arms:
        raise ValueError("serve_parent arm is required (reference for gate deltas)")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if parent_epochs < 1:
        raise ValueError(f"parent_epochs must be >= 1, got {parent_epochs}")
    aid = (account_id or "").strip()
    if not aid:
        raise ValueError("account_id is required")

    ft_arms = [name for name in arms if name != "serve_parent"]
    if ft_arms and epochs < 1:
        raise ValueError("FT arms require epochs >= 1")

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
    if ft_arms and len(user_train) < min_train_moves:
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
        "style_disagree_scale": style_disagree_scale,
        "n_parent_train": len(parent_batch),
        "n_user_train": len(user_train),
        "n_user_val": len(user_val),
        "cold_start_parent": True,
        "test_read": False,
        "serve_gate": {
            "min_disagree_lift": SERVE_GATE_MIN_DISAGREE_LIFT,
            "max_overall_drop": SERVE_GATE_MAX_OVERALL_DROP,
            "max_pawn_gap_regress": SERVE_GATE_MAX_PAWN_GAP_REGRESS,
            "ll_warn": SERVE_GATE_LL_WARN,
        },
        "arms": {},
    }
    logger.info(
        "Serve-gate A/B parent=%s user_train=%s user_val=%s parent_ep=%s warm_ep=%s arms=%s",
        len(parent_batch),
        len(user_train),
        len(user_val),
        parent_epochs,
        epochs,
        arms,
    )

    with tempfile.TemporaryDirectory(prefix="ct-serve-gate-ab-") as tmp:
        parent_path = Path(tmp) / "parent.keras"
        parent_trainer = HybridBoardTrainer(
            epochs=parent_epochs,
            batch_size=batch_size,
            style_disagree_boost=parent_style_disagree_boost,
            style_disagree_scale=style_disagree_scale,
            learning_rate=HybridBoardTrainer.DEFAULT_LEARNING_RATE,
        )
        parent_started = time.monotonic()
        parent_model, parent_metrics = parent_trainer.fit(parent_batch)
        HybridBoardTrainer.save(parent_model, parent_path)
        parent_val, parent_ll = score_hybrid_datums(parent_model, user_val)
        parent_payload = _arm_payload(
            parent_val,
            parent_ll,
            param_count=int(parent_model.count_params()),
            fit_seconds=time.monotonic() - parent_started,
            n_train=int(parent_metrics["n_samples"]),
            train_loss_last=float(parent_metrics.get("loss", float("nan"))),
            extra={
                "arm": "serve_parent",
                "style_disagree_boost": None,
                "served_parent": True,
                "fit_epochs_completed": float(parent_metrics.get("epochs", parent_epochs)),
            },
        )
        report["parent"] = {
            "fit_seconds": parent_payload["fit_seconds"],
            "n_train": parent_payload["n_train"],
            "train_loss_last": parent_payload["train_loss_last"],
            "param_count": parent_payload["param_count"],
            "val": parent_payload["val"],
            "val_log_loss": parent_payload["val_log_loss"],
        }
        if "serve_parent" in arms:
            report["arms"]["serve_parent"] = parent_payload
        del parent_model

        parent_arm = report["arms"].get("serve_parent") or parent_payload
        p_val = parent_arm["val"]
        p_ll = float(parent_arm["val_log_loss"])

        for name in arms:
            if name == "serve_parent":
                continue
            boost = float(_ARM_BOOST[name])  # type: ignore[arg-type]
            logger.info(
                "Running arm=%s style_disagree_boost=%s n_train=%s",
                name,
                boost,
                len(user_train),
            )
            arm = _fit_ft_arm(
                name=name,
                train=user_train,
                val=user_val,
                parent_path=parent_path,
                epochs=epochs,
                batch_size=batch_size,
                style_disagree_boost=boost,
                style_disagree_scale=style_disagree_scale,
            )
            c_val = arm["val"]
            gate = evaluate_serve_gate(
                parent_top1=float(p_val["top1_overall"]),
                parent_disagree=float(p_val.get("top1_sf_disagree", float("nan"))),
                parent_ll=p_ll,
                parent_pawn_gap=float(p_val["sf_delta_mean_pawns"]),
                child_top1=float(c_val["top1_overall"]),
                child_disagree=float(c_val.get("top1_sf_disagree", float("nan"))),
                child_ll=float(arm["val_log_loss"]),
                child_pawn_gap=float(c_val["sf_delta_mean_pawns"]),
            )
            arm["gate"] = gate
            report["arms"][name] = arm

    # Attach trivial gate on serve_parent for table completeness.
    if "serve_parent" in report["arms"]:
        report["arms"]["serve_parent"]["gate"] = {
            "d_top1": 0.0,
            "d_disagree": 0.0,
            "d_ll": 0.0,
            "d_pawn_gap": 0.0,
            "pawn_gap_regress": 0.0,
            "pass_disagree": False,
            "pass_overall": True,
            "pass_pawn": True,
            "ll_warn": False,
            "passes_serve_gate": False,
            "reference": True,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote serve-gate A/B report to %s", output)
    return report


def format_report(report: dict[str, Any]) -> str:
    lines = [
        (
            f"account={str(report.get('account_id', ''))[:12]}… "
            f"parent={report['n_parent_train']} user_train={report['n_user_train']} "
            f"user_val={report['n_user_val']} warm_epochs={report['epochs_warm']}"
        )
    ]
    gate_cfg = report.get("serve_gate") or {}
    lines.append(
        "gate: "
        f"disagree_lift>={float(gate_cfg.get('min_disagree_lift', SERVE_GATE_MIN_DISAGREE_LIFT)):.2f} "
        f"|d_top1|<={float(gate_cfg.get('max_overall_drop', SERVE_GATE_MAX_OVERALL_DROP)):.2f} "
        f"(allow drop) pawn_regress<="
        f"{float(gate_cfg.get('max_pawn_gap_regress', SERVE_GATE_MAX_PAWN_GAP_REGRESS)):.2f} "
        f"ll_warn>{float(gate_cfg.get('ll_warn', SERVE_GATE_LL_WARN)):.2f}"
    )
    for name, arm in (report.get("arms") or {}).items():
        val = arm["val"]
        gate = arm.get("gate") or {}
        boost = arm.get("style_disagree_boost")
        boost_s = "parent" if boost is None else f"{float(boost):.1f}"
        gate_s = ""
        if name != "serve_parent":
            passed = "PASS" if gate.get("passes_serve_gate") else "FAIL"
            gate_s = (
                f" gate={passed}"
                f" d_top1={gate.get('d_top1', float('nan')):+.4f}"
                f" d_disagree={gate.get('d_disagree', float('nan')):+.4f}"
                f" d_ll={gate.get('d_ll', float('nan')):+.4f}"
                f" pawn_reg={gate.get('pawn_gap_regress', float('nan')):+.4f}"
            )
            if gate.get("ll_warn"):
                gate_s += " ll_warn"
        lines.append(
            f"{name}: boost={boost_s} "
            f"top1={val['top1_overall']:.4f} "
            f"agree={val.get('top1_sf_agree', float('nan')):.4f} "
            f"disagree={val.get('top1_sf_disagree', float('nan')):.4f} "
            f"log_loss={arm['val_log_loss']:.4f} "
            f"pawn_gap={val['sf_delta_mean_pawns']:.4f} "
            f"fit_s={arm['fit_seconds']:.1f}{gate_s}"
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
    parser.add_argument("--epochs", type=int, default=2, help="Warm finetune epochs per FT arm.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--arms",
        default="serve_parent,ft_boost_1,ft_boost_2,ft_boost_4",
        help="Comma-separated: serve_parent, ft_boost_1, ft_boost_2, ft_boost_4",
    )
    parser.add_argument(
        "--parent-style-disagree-boost",
        type=float,
        default=DEFAULT_STYLE_DISAGREE_BOOST,
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
        default=Path("storage/tmp/serve_gate_ab.json"),
    )
    args = parser.parse_args()
    arms = [part.strip() for part in str(args.arms).split(",") if part.strip()]
    parent_train_limit = None if args.full_buckets else args.parent_train_limit
    user_train_limit = None if args.full_buckets else args.user_train_limit
    val_limit = None if args.full_buckets else args.val_limit
    report = run_serve_gate_ab(
        account_id=str(args.account_id),
        parent_train_limit=parent_train_limit,
        user_train_limit=user_train_limit,
        val_limit=val_limit,
        parent_epochs=int(args.parent_epochs),
        epochs=int(args.epochs),
        arms=arms,
        batch_size=int(args.batch_size),
        parent_style_disagree_boost=float(args.parent_style_disagree_boost),
        style_disagree_scale=float(args.style_disagree_scale),
        split_version=str(args.split_version),
        output=args.output,
        min_train_moves=int(args.min_train_moves),
    )
    print("\n" + format_report(report))
    return 0
