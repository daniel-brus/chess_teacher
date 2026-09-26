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
    load_balanced_account_registry_bucket_datums,
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
# Packing stacks (n, 128, 55) float32 ≈ n * 28 KiB; ~100k fit on 8GB host, OOM at ~200k.
DEFAULT_FIT_CHUNK_SIZE = 40_000


def _fit_hybrid_in_chunks(
    trainer: HybridBoardTrainer,
    datums: list[Any],
    *,
    out_path: Path,
    chunk_size: int,
    initial_weights: Path | None = None,
    require_parent_weights: bool = False,
    recency_lambda: float | None = None,
    balanced_halves: bool = False,
) -> tuple[Any, dict[str, float]]:
    """Fit hybrid model in memory-safe chunks; warm-start across chunks.

    When ``balanced_halves`` is True, ``datums`` is treated as
    ``[account_a...][account_b...]`` with equal lengths (or nearly); each chunk
    takes the next slice from both halves so mix stays ~50/50.
    """
    n = len(datums)
    chunk_size = max(MIN_USER_TRAIN_MOVES, int(chunk_size))
    weights: Path | None = initial_weights
    model: Any = None
    metrics: dict[str, float] = {}

    if balanced_halves and n >= 2:
        mid = n // 2
        left, right = datums[:mid], datums[mid:]
        per = max(MIN_USER_TRAIN_MOVES // 2, chunk_size // 2)
        starts = list(range(0, max(len(left), len(right)), per))
        if not starts:
            starts = [0]
        logger.info(
            "Chunked balanced hybrid fit n=%s halves=%s/%s per_side=%s chunks=%s → %s",
            n,
            len(left),
            len(right),
            per,
            len(starts),
            out_path,
        )
        for i, start in enumerate(starts, start=1):
            chunk = left[start : start + per] + right[start : start + per]
            if len(chunk) < MIN_USER_TRAIN_MOVES:
                continue
            logger.info(
                "Hybrid chunk %s/%s n=%s weights=%s",
                i,
                len(starts),
                len(chunk),
                weights,
            )
            model, metrics = trainer.fit(
                chunk,
                weights_path=weights,
                require_parent_weights=weights is not None,
                recency_lambda=recency_lambda,
            )
            HybridBoardTrainer.save(model, out_path)
            weights = out_path
    else:
        starts = list(range(0, n, chunk_size))
        logger.info(
            "Chunked hybrid fit n=%s chunk_size=%s chunks=%s → %s",
            n,
            chunk_size,
            len(starts),
            out_path,
        )
        for i, start in enumerate(starts, start=1):
            chunk = datums[start : start + chunk_size]
            if len(chunk) < MIN_USER_TRAIN_MOVES:
                continue
            logger.info(
                "Hybrid chunk %s/%s n=%s weights=%s",
                i,
                len(starts),
                len(chunk),
                weights,
            )
            model, metrics = trainer.fit(
                chunk,
                weights_path=weights,
                require_parent_weights=weights is not None,
                recency_lambda=recency_lambda,
            )
            HybridBoardTrainer.save(model, out_path)
            weights = out_path

    if model is None:
        raise RuntimeError("Chunked hybrid fit produced no model")
    return model, metrics


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
    balance_account_ids: list[str] | None = None,
    fit_chunk_size: int | None = None,
) -> Any:
    """Cold-start hybrid on registry train; save to ``parent_out``.

    Default: all-accounts registry-train prefix (``game_id`` ASC).

    With ``balance_account_ids`` (>=2): equal per-account prefixes of size
    ``train_limit`` each (required), concatenated — true N-way balance, still
    hash-registry only.

    ``fit_chunk_size``: if set (or auto when n is large), fit in warm-start
    chunks to avoid packing OOM on big parents.
    """
    balanced = [a.strip() for a in (balance_account_ids or []) if a.strip()]
    if balanced:
        if train_limit is None or train_limit < 1:
            raise RuntimeError(
                "Balanced parent requires --parent-train-limit as per-account "
                "move cap (e.g. 25000 for 25k+25k)."
            )
        logger.info(
            "Loading balanced registry train split_version=%s per_account_limit=%s accounts=%s…",
            split_version,
            train_limit,
            balanced,
        )
        train = load_balanced_account_registry_bucket_datums(
            balanced,
            bucket=SplitBucket.TRAIN,
            split_version=split_version,
            per_account_limit=train_limit,
        )
        # Rough mix log (account_id on TrainingDatum when present).
        counts: dict[str, int] = {}
        for d in train:
            aid = str(getattr(d, "account_id", "") or "?")
            counts[aid] = counts.get(aid, 0) + 1
        logger.info("Balanced parent mix moves_by_account=%s total=%s", counts, len(train))
    else:
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
    chunk = fit_chunk_size
    if chunk is None and len(train) > DEFAULT_FIT_CHUNK_SIZE:
        chunk = DEFAULT_FIT_CHUNK_SIZE
    logger.info(
        "Cold hybrid parent fit n=%s epochs=%s style_disagree_boost=%s fit_chunk_size=%s → %s",
        len(train),
        epochs,
        style_disagree_boost,
        chunk,
        parent_out,
    )
    t0 = time.perf_counter()
    parent_out.parent.mkdir(parents=True, exist_ok=True)
    if chunk is not None and len(train) > chunk:
        model, metrics = _fit_hybrid_in_chunks(
            trainer,
            train,
            out_path=parent_out,
            chunk_size=chunk,
            balanced_halves=bool(balanced),
        )
    else:
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
    parent_balance_account_ids: list[str] | None = None,
    parent_fit_chunk_size: int | None = None,
    fit_chunk_size: int | None = None,
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
    if parent_balance_account_ids and not train_parent:
        logger.error("--parent-balance-account-ids only valid with --train-parent.")
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
                balance_account_ids=parent_balance_account_ids,
                fit_chunk_size=parent_fit_chunk_size,
            )
        except (RuntimeError, MemoryError) as exc:
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
    child_path = (
        Path(child_out)
        if child_out
        else parent_path.with_name(parent_path.stem + f"_{account_id[:8]}_ft.keras")
    )
    child_path.parent.mkdir(parents=True, exist_ok=True)
    user_chunk = fit_chunk_size
    if user_chunk is None and len(train) > DEFAULT_FIT_CHUNK_SIZE:
        user_chunk = DEFAULT_FIT_CHUNK_SIZE
    try:
        if user_chunk is not None and len(train) > user_chunk:
            child_model, train_metrics = _fit_hybrid_in_chunks(
                trainer,
                train,
                out_path=child_path,
                chunk_size=user_chunk,
                initial_weights=parent_path,
                require_parent_weights=True,
                recency_lambda=recency_lambda,
                balanced_halves=False,
            )
        else:
            child_model, train_metrics = trainer.fit(
                train,
                weights_path=parent_path,
                require_parent_weights=True,
                recency_lambda=recency_lambda,
            )
            if child_out:
                HybridBoardTrainer.save(child_model, child_path)
    except MemoryError as exc:
        logger.error("User finetune OOM: %s", exc)
        return 1
    fit_s = time.perf_counter() - t0
    logger.info(
        "User fit done in %.1fs train_top1=%.4f",
        fit_s,
        train_metrics.get("masked_cand_top1", 0.0),
    )
    if child_out and not child_path.is_file():
        HybridBoardTrainer.save(child_model, child_path)

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
        help=(
            "Complete-game move cap for cold parent train (None = full registry "
            "train). With --parent-balance-account-ids: per-account cap "
            "(e.g. 25000 → 25k+25k)."
        ),
    )
    parser.add_argument(
        "--parent-balance-account-ids",
        type=str,
        default=None,
        help=(
            "Comma-separated account_ids for equal per-account parent mix "
            "(requires --train-parent + --parent-train-limit per account)."
        ),
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
        "--parent-fit-chunk-size",
        type=int,
        default=None,
        help=(
            "Pack/fit parent in warm-start chunks of this many moves "
            f"(default: auto {DEFAULT_FIT_CHUNK_SIZE} when n is larger)."
        ),
    )
    parser.add_argument(
        "--fit-chunk-size",
        type=int,
        default=None,
        help=(
            "Pack/fit user FT in warm-start chunks "
            f"(default: auto {DEFAULT_FIT_CHUNK_SIZE} when train n is larger)."
        ),
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
    balance_raw = str(args.parent_balance_account_ids or "").strip()
    balance_ids = [p.strip() for p in balance_raw.split(",") if p.strip()] or None
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
        parent_balance_account_ids=balance_ids,
        parent_fit_chunk_size=int(args.parent_fit_chunk_size)
        if args.parent_fit_chunk_size is not None
        else None,
        fit_chunk_size=int(args.fit_chunk_size) if args.fit_chunk_size is not None else None,
        train_limit=int(args.train_limit) if args.train_limit is not None else None,
        val_limit=int(args.val_limit) if args.val_limit is not None else None,
        epochs=max(1, int(args.epochs)),
        style_disagree_boost=float(args.style_disagree_boost),
        style_disagree_scale=float(args.style_disagree_scale),
        recency_lambda=float(args.recency_lambda) if args.recency_lambda is not None else None,
        child_out=str(args.child_out) if args.child_out else None,
        min_train_moves=max(1, int(args.min_train_moves)),
    )
