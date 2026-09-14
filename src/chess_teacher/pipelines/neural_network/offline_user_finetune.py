"""Phase 3a: offline user finetune on account ∩ hash-registry train/val.

Parent = hybrid baseline (cold-trained on all-accounts registry train, or a
local ``.keras`` path). Child trains on that account's registry-train moves;
eval is the same account's registry-val. Primary metric: ``top1_sf_disagree``.
Agree is reported only (no hard fail). Never writes production tables.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from chess_teacher.pipelines.neural_network.board_encoder import (
    HybridBoardTrainer,
    load_hybrid_board_keras,
)
from chess_teacher.pipelines.neural_network.eval_metrics import (
    evaluate_datums,
    format_eval_delta,
    format_eval_metrics,
)
from chess_teacher.pipelines.neural_network.offline_eval import (
    load_account_registry_split,
    load_registry_bucket_datums,
)
from chess_teacher.pipelines.neural_network.splits import (
    DEFAULT_SPLIT_SALT,
    SplitBucket,
    format_split_summary,
)
from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.logging import get_logger

logger = get_logger()

# Roadmap Phase 3a min-data gate: skip / serve baseline below this.
MIN_USER_TRAIN_MOVES = 300
DEFAULT_USER_STYLE_DISAGREE_BOOST = 4.0


def _resolve_parent_path(
    *,
    parent_weights: str | None,
    train_parent: bool,
    parent_out: str | None,
) -> Path | None:
    if train_parent:
        if not parent_out:
            raise ValueError("--train-parent requires --parent-out")
        return Path(parent_out)
    if not parent_weights:
        return None
    return Path(parent_weights)


def train_cold_hybrid_parent(
    *,
    split_version: str,
    train_limit: int | None,
    epochs: int,
    style_disagree_boost: float,
    style_disagree_scale: float,
    parent_out: Path,
) -> Any:
    """Cold-start hybrid on all-accounts registry train; save to ``parent_out``."""
    logger.info(
        "Loading all-accounts registry train split_version=%s limit=%s…",
        split_version,
        train_limit,
    )
    train = load_registry_bucket_datums(
        bucket=SplitBucket.TRAIN,
        split_version=split_version,
        limit=train_limit,
    )
    if len(train) < MIN_USER_TRAIN_MOVES:
        raise RuntimeError(
            f"Platform registry-train too small for parent: {len(train)} moves "
            f"(need >={MIN_USER_TRAIN_MOVES}). Backfill splits / raise --parent-train-limit."
        )
    trainer = HybridBoardTrainer(
        epochs=epochs,
        style_disagree_boost=style_disagree_boost,
        style_disagree_scale=style_disagree_scale,
    )
    logger.info(
        "Cold hybrid parent fit n=%s epochs=%s style_disagree_boost=%s → %s",
        len(train),
        epochs,
        style_disagree_boost,
        parent_out,
    )
    t0 = time.perf_counter()
    model, metrics = trainer.fit(train)
    HybridBoardTrainer.save(model, parent_out)
    logger.info(
        "Parent fit done in %.1fs train_top1=%.4f path=%s",
        time.perf_counter() - t0,
        metrics.get("masked_cand_top1", 0.0),
        parent_out,
    )
    return model


def run_offline_user_finetune_eval(
    *,
    account_id: str | None,
    split_version: str,
    parent_weights: str | None,
    train_parent: bool,
    parent_out: str | None,
    parent_train_limit: int | None,
    parent_epochs: int,
    parent_style_disagree_boost: float,
    train_limit: int | None,
    val_limit: int | None,
    epochs: int,
    style_disagree_boost: float,
    style_disagree_scale: float,
    recency_lambda: float | None,
    child_out: str | None,
    min_train_moves: int = MIN_USER_TRAIN_MOVES,
) -> int:
    """Train optional cold parent, finetune one account, print stratified deltas."""
    try:
        parent_path = _resolve_parent_path(
            parent_weights=parent_weights,
            train_parent=train_parent,
            parent_out=parent_out,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    if train_parent and parent_weights:
        logger.error("Pass exactly one of --parent-weights or --train-parent.")
        return 1
    if not train_parent and not parent_weights:
        logger.error("Pass --parent-weights or --train-parent.")
        return 1
    if parent_path is None:
        logger.error("Parent weights path unresolved.")
        return 1

    db = get_db_client()

    if train_parent:
        try:
            train_cold_hybrid_parent(
                split_version=split_version,
                train_limit=parent_train_limit,
                epochs=parent_epochs,
                style_disagree_boost=parent_style_disagree_boost,
                style_disagree_scale=style_disagree_scale,
                parent_out=parent_path,
            )
        except RuntimeError as exc:
            logger.error("%s", exc)
            return 1

    if not account_id:
        if train_parent:
            print(f"\nparent_only=true parent_weights={parent_path}")
            return 0
        logger.error("--account-id is required unless --train-parent without finetune.")
        return 1

    if not parent_path.is_file():
        logger.error("Parent weights missing: %s", parent_path)
        return 1

    split = load_account_registry_split(
        account_id,
        db,
        split_version=split_version,
        train_limit=train_limit,
        val_limit=val_limit,
    )
    print(
        "\n"
        + format_split_summary(
            split,
            heading=f"account x registry split account_id={account_id!r}",
        )
    )
    train = split.train_datums
    val = split.val_datums
    if len(train) < min_train_moves:
        logger.error(
            "User registry-train below min-data gate: n=%s < %s — skip / serve baseline.",
            len(train),
            min_train_moves,
        )
        return 2
    if len(val) < 10:
        logger.error("User registry-val too small: %s moves", len(val))
        return 1

    logger.info("Loading parent hybrid from %s", parent_path)
    parent_model = load_hybrid_board_keras(parent_path, compile_model=False)
    parent_metrics = evaluate_datums(parent_model, val)

    trainer = HybridBoardTrainer(
        epochs=epochs,
        style_disagree_boost=style_disagree_boost,
        style_disagree_scale=style_disagree_scale,
    )
    logger.info(
        "User finetune account=%s train_n=%s val_n=%s epochs=%s "
        "style_disagree_boost=%s recency_lambda=%s",
        account_id,
        len(train),
        len(val),
        epochs,
        style_disagree_boost,
        recency_lambda,
    )
    t0 = time.perf_counter()
    child_model, train_metrics = trainer.fit(
        train,
        weights_path=parent_path,
        require_parent_weights=True,
        recency_lambda=recency_lambda,
    )
    fit_s = time.perf_counter() - t0
    logger.info(
        "User fit done in %.1fs train_top1=%.4f",
        fit_s,
        train_metrics.get("masked_cand_top1", 0.0),
    )
    if child_out:
        HybridBoardTrainer.save(child_model, Path(child_out))

    child_metrics = evaluate_datums(child_model, val)

    print("\n=== offline user finetune eval (no DB write) ===")
    print(f"account_id={account_id!r} split_version={split_version!r}")
    print(f"parent_weights={parent_path}")
    print(f"train_n={len(train)} val_n={len(val)} fit_s={fit_s:.1f}")
    print(format_eval_metrics("parent_user_val", parent_metrics))
    print(format_eval_metrics("user_ft_user_val", child_metrics))
    print(
        format_eval_delta(
            child_metrics,
            parent_metrics,
            candidate_name="user_ft",
            baseline_name="parent",
        )
    )
    print("primary=top1_sf_disagree  agree=report_only  no_promote=true")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--account-id",
        type=str,
        default=None,
        help="Platform account_id. Optional with --train-parent alone (parent-only).",
    )
    parser.add_argument("--split-version", type=str, default=DEFAULT_SPLIT_SALT)
    parent = parser.add_mutually_exclusive_group(required=True)
    parent.add_argument(
        "--parent-weights",
        type=str,
        default=None,
        help="Local hybrid .keras parent (required unless --train-parent).",
    )
    parent.add_argument(
        "--train-parent",
        action="store_true",
        help="Cold-start hybrid on all-accounts registry train, write --parent-out.",
    )
    parser.add_argument(
        "--parent-out",
        type=str,
        default=None,
        help="Where to write cold parent (.keras). Required with --train-parent.",
    )
    parser.add_argument(
        "--parent-train-limit",
        type=int,
        default=None,
        help="Complete-game move cap for cold parent train (None = full registry train).",
    )
    parser.add_argument(
        "--parent-epochs",
        type=int,
        default=HybridBoardTrainer.DEFAULT_EPOCHS,
    )
    parser.add_argument(
        "--parent-style-disagree-boost",
        type=float,
        default=2.0,
        help="Style boost for cold parent only (platform-ish; default 2.0).",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Complete-game move cap for user∩registry-train (None = all).",
    )
    parser.add_argument(
        "--val-limit",
        type=int,
        default=None,
        help="Complete-game move cap for user∩registry-val (None = all).",
    )
    parser.add_argument("--epochs", type=int, default=HybridBoardTrainer.DEFAULT_EPOCHS)
    parser.add_argument(
        "--style-disagree-boost",
        type=float,
        default=DEFAULT_USER_STYLE_DISAGREE_BOOST,
    )
    parser.add_argument("--style-disagree-scale", type=float, default=2.0)
    parser.add_argument(
        "--recency-lambda",
        type=float,
        default=None,
        help="Optional recency sample-weight lambda (weights only; never chooses val).",
    )
    parser.add_argument(
        "--child-out",
        type=str,
        default=None,
        help="Optional path to save finetuned user .keras.",
    )
    parser.add_argument(
        "--min-train-moves",
        type=int,
        default=MIN_USER_TRAIN_MOVES,
        help="Min user∩registry-train moves before finetune (default 300).",
    )
    args = parser.parse_args()
    return run_offline_user_finetune_eval(
        account_id=str(args.account_id).strip() if args.account_id else None,
        split_version=str(args.split_version),
        parent_weights=str(args.parent_weights) if args.parent_weights else None,
        train_parent=bool(args.train_parent),
        parent_out=str(args.parent_out) if args.parent_out else None,
        parent_train_limit=int(args.parent_train_limit)
        if args.parent_train_limit is not None
        else None,
        parent_epochs=max(1, int(args.parent_epochs)),
        parent_style_disagree_boost=float(args.parent_style_disagree_boost),
        train_limit=int(args.train_limit) if args.train_limit is not None else None,
        val_limit=int(args.val_limit) if args.val_limit is not None else None,
        epochs=max(1, int(args.epochs)),
        style_disagree_boost=float(args.style_disagree_boost),
        style_disagree_scale=float(args.style_disagree_scale),
        recency_lambda=float(args.recency_lambda) if args.recency_lambda is not None else None,
        child_out=str(args.child_out) if args.child_out else None,
        min_train_moves=max(1, int(args.min_train_moves)),
    )
