"""Offline A/B: inference logit blend parent↔child (Phase 5 Slice E).

Train one shared cold parent + one user FT child (default boost 4 - today's
aggressive personal path that failed Slice D serve gate). Score parent, child,
and logit blends ``(1-alpha)*parent + alpha*child`` on account x registry-val.

Hypothesis: blend recovers overall/pawn strength while keeping some disagree
lift, so an alpha in (0,1) can pass the Slice D serve gate where pure child fails.

Arms
----
``serve_parent`` (alpha=0), ``blend_0.25``, ``blend_0.50``, ``blend_0.75``,
``serve_child`` (alpha=1).

Smoke::

    .venv/Scripts/python.exe scripts/tools/offline_inference_blend_ab.py \\
      --parent-train-limit 2000 --user-train-limit 1500 --val-limit 500 \\
      --parent-epochs 2 --epochs 2 \\
      --output storage/tmp/inference_blend_ab_smoke.json

Mid decision::

    .venv/Scripts/python.exe scripts/tools/offline_inference_blend_ab.py \\
      --parent-train-limit 30000 --user-train-limit 15000 --val-limit 4000 \\
      --parent-epochs 3 --epochs 3 \\
      --output storage/tmp/inference_blend_ab_mid.json
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from chess_teacher.pipelines.neural_network.board_encoder import (
    HybridBoardTrainer,
    load_hybrid_board_keras,
)
from chess_teacher.pipelines.neural_network.hybrid_val_score import score_hybrid_logit_blend
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

_DEFAULT_ACCOUNT_ID = "5c3f069730e098aedc61b978c92d6448eebfdf234e575552e86343a10bfb698b"
_ARMS = ("serve_parent", "blend_0.25", "blend_0.50", "blend_0.75", "serve_child")
_ARM_ALPHA = {
    "serve_parent": 0.0,
    "blend_0.25": 0.25,
    "blend_0.50": 0.50,
    "blend_0.75": 0.75,
    "serve_child": 1.0,
}


def run_inference_blend_ab(
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
    if "serve_parent" not in arms:
        raise ValueError("serve_parent arm is required (gate reference)")
    if epochs < 1 or parent_epochs < 1:
        raise ValueError("epochs and parent_epochs must be >= 1")
    aid = (account_id or "").strip()
    if not aid:
        raise ValueError("account_id is required")

    ensure_tensorflow_logging()
    db = get_db_client()
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
        "blend_kind": "logit_linear",
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
        "Inference-blend A/B parent=%s user_train=%s user_val=%s child_boost=%s arms=%s",
        len(parent_batch),
        len(user_train),
        len(user_val),
        user_style_disagree_boost,
        arms,
    )

    with tempfile.TemporaryDirectory(prefix="ct-inference-blend-ab-") as tmp:
        parent_path = Path(tmp) / "parent.keras"
        child_path = Path(tmp) / "child.keras"

        parent_trainer = HybridBoardTrainer(
            epochs=parent_epochs,
            batch_size=batch_size,
            style_disagree_boost=parent_style_disagree_boost,
            style_disagree_scale=style_disagree_scale,
            learning_rate=HybridBoardTrainer.DEFAULT_LEARNING_RATE,
        )
        t0 = time.monotonic()
        parent_model, parent_metrics = parent_trainer.fit(parent_batch)
        HybridBoardTrainer.save(parent_model, parent_path)
        report["parent_fit_seconds"] = time.monotonic() - t0
        report["parent_n_train"] = int(parent_metrics["n_samples"])
        del parent_model

        child_trainer = HybridBoardTrainer(
            epochs=epochs,
            batch_size=batch_size,
            style_disagree_boost=user_style_disagree_boost,
            style_disagree_scale=style_disagree_scale,
            learning_rate=HybridBoardTrainer.DEFAULT_LEARNING_RATE,
            warm_learning_rate=HybridBoardTrainer.DEFAULT_LEARNING_RATE,
            warm_warmup_epochs=0,
        )
        t1 = time.monotonic()
        child_model, child_metrics = child_trainer.fit(
            user_train,
            weights_path=parent_path,
            require_parent_weights=True,
        )
        HybridBoardTrainer.save(child_model, child_path)
        report["child_fit_seconds"] = time.monotonic() - t1
        report["child_n_train"] = int(child_metrics["n_samples"])
        del child_model

        parent_model = load_hybrid_board_keras(parent_path, compile_model=False)
        child_model = load_hybrid_board_keras(child_path, compile_model=False)

        parent_ref: dict[str, Any] | None = None
        for name in arms:
            alpha = float(_ARM_ALPHA[name])
            logger.info("Scoring arm=%s alpha=%.2f", name, alpha)
            started = time.monotonic()
            metrics, log_loss = score_hybrid_logit_blend(
                parent_model,
                child_model,
                user_val,
                alpha=alpha,
            )
            payload: dict[str, Any] = {
                "arm": name,
                "alpha": alpha,
                "val": metrics.as_dict(),
                "val_log_loss": log_loss,
                "fit_seconds": time.monotonic() - started,
                "n_train": 0 if alpha == 0.0 else int(child_metrics["n_samples"]),
            }
            if name == "serve_parent":
                parent_ref = payload
                payload["gate"] = {
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
            report["arms"][name] = payload

        if parent_ref is None:
            raise RuntimeError("serve_parent missing after scoring")
        p_val = parent_ref["val"]
        p_ll = float(parent_ref["val_log_loss"])
        for name, arm in report["arms"].items():
            if name == "serve_parent":
                continue
            c_val = arm["val"]
            arm["gate"] = evaluate_serve_gate(
                parent_top1=float(p_val["top1_overall"]),
                parent_disagree=float(p_val.get("top1_sf_disagree", float("nan"))),
                parent_ll=p_ll,
                parent_pawn_gap=float(p_val["sf_delta_mean_pawns"]),
                child_top1=float(c_val["top1_overall"]),
                child_disagree=float(c_val.get("top1_sf_disagree", float("nan"))),
                child_ll=float(arm["val_log_loss"]),
                child_pawn_gap=float(c_val["sf_delta_mean_pawns"]),
            )

        del parent_model, child_model

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote inference-blend A/B report to %s", output)
    return report


def format_report(report: dict[str, Any]) -> str:
    lines = [
        (
            f"account={str(report.get('account_id', ''))[:12]}… "
            f"parent={report['n_parent_train']} user_train={report['n_user_train']} "
            f"user_val={report['n_user_val']} child_boost={report['user_style_disagree_boost']}"
        )
    ]
    for name, arm in (report.get("arms") or {}).items():
        val = arm["val"]
        gate = arm.get("gate") or {}
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
            f"{name}: alpha={float(arm.get('alpha', float('nan'))):.2f} "
            f"top1={val['top1_overall']:.4f} "
            f"agree={val.get('top1_sf_agree', float('nan')):.4f} "
            f"disagree={val.get('top1_sf_disagree', float('nan')):.4f} "
            f"log_loss={arm['val_log_loss']:.4f} "
            f"pawn_gap={val['sf_delta_mean_pawns']:.4f} "
            f"score_s={arm['fit_seconds']:.1f}{gate_s}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account-id", type=str, default=_DEFAULT_ACCOUNT_ID)
    parser.add_argument("--split-version", type=str, default=DEFAULT_SPLIT_SALT)
    parser.add_argument("--parent-train-limit", type=int, default=2000)
    parser.add_argument("--user-train-limit", type=int, default=1500)
    parser.add_argument("--val-limit", type=int, default=500)
    parser.add_argument("--full-buckets", action="store_true")
    parser.add_argument("--parent-epochs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--arms",
        default="serve_parent,blend_0.25,blend_0.50,blend_0.75,serve_child",
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
    parser.add_argument("--min-train-moves", type=int, default=MIN_USER_TRAIN_MOVES)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("storage/tmp/inference_blend_ab.json"),
    )
    args = parser.parse_args()
    arms = [part.strip() for part in str(args.arms).split(",") if part.strip()]
    report = run_inference_blend_ab(
        account_id=str(args.account_id),
        parent_train_limit=None if args.full_buckets else args.parent_train_limit,
        user_train_limit=None if args.full_buckets else args.user_train_limit,
        val_limit=None if args.full_buckets else args.val_limit,
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
