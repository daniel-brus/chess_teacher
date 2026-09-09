"""Offline personal-bot siblings: user hash-split finetune, promotion score, catch-up.

Never writes production training bookkeeping or baseline model rows. One-shot
finetune / promotion still hash-split per account. Catch-up uses the personal
registry queue (linked-account train, ``already_processed_personal``).
"""

from __future__ import annotations

import argparse
import tempfile
import time
from collections.abc import Sequence
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

from chess_teacher.pipelines.neural_network.create_training_set import (
    TrainingDataStore,
    TrainingDatum,
    account_id_in_sql,
)
from chess_teacher.pipelines.neural_network.eval_metrics import (
    EvalMetrics,
    evaluate_datums,
    evaluate_model_uri,
    format_eval_delta,
    format_eval_metrics,
)
from chess_teacher.pipelines.neural_network.mlflow_utils import MLflowTracker
from chess_teacher.pipelines.neural_network.models import (
    PROCESSED_FLAG_PERSONAL,
    BaselineModel,
    BaselineModelStatus,
)
from chess_teacher.pipelines.neural_network.ply_weights import (
    DEFAULT_RECENCY_BOOST,
    DEFAULT_RECENCY_LAMBDA,
    DEFAULT_STYLE_DISAGREE_BOOST,
    DEFAULT_STYLE_DISAGREE_SCALE,
    USER_BASELINE_DISAGREE_BOOST,
)
from chess_teacher.pipelines.neural_network.split_registry import get_split_registry
from chess_teacher.pipelines.neural_network.splits import DEFAULT_SPLIT_SALT, format_split_summary
from chess_teacher.pipelines.neural_network.train import BaselineTrainer
from chess_teacher.pipelines.neural_network.user_splits import (
    DEFAULT_MIN_USER_TRAIN_MOVES,
    DEFAULT_MIN_USER_VAL_MOVES,
    load_account_split,
)
from chess_teacher.platform.account import Account
from chess_teacher.platform.user import User
from chess_teacher.platform.user_account import UserAccount
from chess_teacher.utils.db.client import DatabaseClient, get_db_client
from chess_teacher.utils.general_utils import generate_ident_is_literal
from chess_teacher.utils.logging import get_logger

logger = get_logger()

DEFAULT_USER_RECENCY_LAMBDA = DEFAULT_RECENCY_LAMBDA
DEFAULT_USER_RECENCY_BOOST = DEFAULT_RECENCY_BOOST
DEFAULT_USER_STYLE_DISAGREE_BOOST = DEFAULT_STYLE_DISAGREE_BOOST
DEFAULT_USER_STYLE_DISAGREE_SCALE = DEFAULT_STYLE_DISAGREE_SCALE
DEFAULT_USER_BASELINE_DISAGREE_BOOST = USER_BASELINE_DISAGREE_BOOST
DEFAULT_MIN_USER_NEW_MOVES = 50
DEFAULT_USER_BATCH_LIMIT = 1000
_USER_SPLIT_HEADING = "user hash split (baseline-v1 85/10/5)"


def _keras_parent_path(parent_uri: str | None) -> Path | None:
    if not parent_uri:
        return None
    path = MLflowTracker().download_keras_weights(parent_uri)
    if path is None or not path.is_file():
        raise FileNotFoundError(f"Could not resolve parent Keras weights from uri={parent_uri!r}")
    return path


def resolve_user_parent_uri(db: Any, parent_uri: str | None) -> str:
    """Explicit URI, else latest production baseline artifact URI."""
    if parent_uri:
        return str(parent_uri)
    row = BaselineModel.latest_with_status(db, BaselineModelStatus.PRODUCTION)
    if row is None or not row.model_uri:
        raise RuntimeError(
            "No production baseline model_uri in ml.baseline_models; pass --parent-uri explicitly."
        )
    return row.model_uri


def _print_val_curve(rows: list[tuple[int, str, int, EvalMetrics]]) -> None:
    print("\n=== offline user catch-up val curve (frozen val, no DB write) ===")
    if not rows:
        print("no rounds")
        return
    for round_i, last_id, n_train, metrics in rows:
        print(
            f"round={round_i} last_game_id={last_id} train_n={n_train} "
            f"{format_eval_metrics('val', metrics)}"
        )


def _print_compare(
    *,
    candidate: EvalMetrics,
    parent: EvalMetrics,
    candidate_label: str,
    parent_uri: str,
    val_n: int,
    extra_lines: Sequence[str] = (),
) -> None:
    print("\n=== offline user compare (no DB write) ===")
    print(f"val_n={val_n}")
    print(f"parent_uri={parent_uri}")
    for line in extra_lines:
        print(line)
    print(format_eval_metrics(candidate_label, candidate))
    print(format_eval_metrics("parent", parent))
    print(
        format_eval_delta(
            candidate,
            parent,
            candidate_name=candidate_label,
            baseline_name="parent",
        )
    )
    print("primary=top1_sf_disagree")
    print("secondary=top3_sf_disagree")
    print("agree_guardrail=agree_t1_drop_le_0.03")
    print("no_promote=true")


def _make_user_trainer(
    *,
    epochs: int,
    style_disagree_boost: float,
    style_disagree_scale: float,
    baseline_disagree_boost: float,
    recency_boost: float,
) -> BaselineTrainer:
    return BaselineTrainer(
        epochs=epochs,
        style_disagree_boost=style_disagree_boost,
        style_disagree_scale=style_disagree_scale,
        baseline_disagree_boost=baseline_disagree_boost,
        recency_boost=recency_boost,
    )


def _load_user_split(
    store: TrainingDataStore,
    account_id: str,
    *,
    limit: int | None,
) -> tuple[Any, dict[str, datetime]]:
    return load_account_split(store, account_id, limit=limit)


def linked_account_ids_for_account(db: DatabaseClient, account_id: str) -> list[str]:
    """CLI ``account_id`` -> that user's linked accounts (user-level queue)."""
    account = Account.fetch_from_db(db, id=account_id)
    links = UserAccount.fetch_all_from_db(
        db,
        where=generate_ident_is_literal("account_id", account.account_id),
    )
    seen: dict[str, None] = {}
    for link in links:
        user = User.fetch_from_db(db, id=link.user_id)
        for linked in user.get_linked_accounts(db):
            if linked.account_id:
                seen[linked.account_id] = None
    return list(seen)


def _score_parent(parent_uri: str, val: list[TrainingDatum]) -> EvalMetrics:
    return evaluate_model_uri(parent_uri, val)


def run_offline_user_finetune_eval(
    *,
    account_id: str,
    parent_uri: str | None,
    recency_lambda: float = DEFAULT_USER_RECENCY_LAMBDA,
    recency_boost: float = DEFAULT_USER_RECENCY_BOOST,
    style_disagree_boost: float = DEFAULT_USER_STYLE_DISAGREE_BOOST,
    style_disagree_scale: float = DEFAULT_USER_STYLE_DISAGREE_SCALE,
    baseline_disagree_boost: float = DEFAULT_USER_BASELINE_DISAGREE_BOOST,
    epochs: int = BaselineTrainer.DEFAULT_EPOCHS,
    min_train_moves: int = DEFAULT_MIN_USER_TRAIN_MOVES,
    limit: int | None = None,
) -> int:
    db = get_db_client()
    store = TrainingDataStore(db)
    split, end_times = _load_user_split(store, account_id, limit=limit)
    print("\n" + format_split_summary(split, heading=_USER_SPLIT_HEADING))
    train = split.train_datums
    val = split.val_datums

    try:
        resolved_parent = resolve_user_parent_uri(db, parent_uri)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    if len(train) < min_train_moves:
        logger.info(
            "Skip user finetune: train_n=%s < min_train_moves=%s",
            len(train),
            min_train_moves,
        )
        if len(val) < DEFAULT_MIN_USER_VAL_MOVES:
            logger.error("Val split too small: %s moves", len(val))
            return 1
        parent_metrics = _score_parent(resolved_parent, val)
        print(format_eval_metrics("parent", parent_metrics))
        print("skipped_train=true")
        print("primary=top1_sf_disagree")
        print("no_promote=true")
        return 0

    if len(val) < DEFAULT_MIN_USER_VAL_MOVES:
        logger.error("Val split too small: %s moves", len(val))
        return 1

    try:
        weights_path = _keras_parent_path(resolved_parent)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    trainer = _make_user_trainer(
        epochs=epochs,
        style_disagree_boost=style_disagree_boost,
        style_disagree_scale=style_disagree_scale,
        baseline_disagree_boost=baseline_disagree_boost,
        recency_boost=recency_boost,
    )
    logger.info(
        "Finetuning user account=%s train_n=%s val_n=%s parent=%s recency_lambda=%s",
        account_id,
        len(train),
        len(val),
        resolved_parent,
        recency_lambda,
    )
    t0 = time.perf_counter()
    try:
        model, _train_metrics = trainer.fit(
            train,
            weights_path=weights_path,
            baseline_weights_path=weights_path,
            recency_lambda=recency_lambda,
            end_time_by_game_id=end_times,
            require_parent_weights=True,
        )
    except (RuntimeError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 1
    logger.info("User finetune fit done in %.1fs", time.perf_counter() - t0)
    candidate_metrics = evaluate_datums(model, val)
    parent_metrics = _score_parent(resolved_parent, val)
    _print_compare(
        candidate=candidate_metrics,
        parent=parent_metrics,
        candidate_label=f"finetune@{epochs}ep",
        parent_uri=resolved_parent,
        val_n=len(val),
    )
    return 0


def run_offline_user_promotion(
    *,
    account_id: str,
    candidate_uri: str | None,
    train_inline: bool,
    parent_uri: str | None,
    recency_lambda: float = DEFAULT_USER_RECENCY_LAMBDA,
    recency_boost: float = DEFAULT_USER_RECENCY_BOOST,
    style_disagree_boost: float = DEFAULT_USER_STYLE_DISAGREE_BOOST,
    style_disagree_scale: float = DEFAULT_USER_STYLE_DISAGREE_SCALE,
    baseline_disagree_boost: float = DEFAULT_USER_BASELINE_DISAGREE_BOOST,
    epochs: int = BaselineTrainer.DEFAULT_EPOCHS,
    min_train_moves: int = DEFAULT_MIN_USER_TRAIN_MOVES,
    limit: int | None = None,
) -> int:
    if train_inline == bool(candidate_uri):
        logger.error("Pass exactly one of --candidate-uri or --train-inline.")
        return 1

    db = get_db_client()
    store = TrainingDataStore(db)
    split, end_times = _load_user_split(store, account_id, limit=limit)
    print("\n" + format_split_summary(split, heading=_USER_SPLIT_HEADING))
    train = split.train_datums
    val = split.val_datums
    if len(val) < DEFAULT_MIN_USER_VAL_MOVES:
        logger.error("Val split too small: %s moves", len(val))
        return 1

    try:
        resolved_parent = resolve_user_parent_uri(db, parent_uri)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    extra: list[str] = []
    if train_inline:
        if len(train) < min_train_moves:
            logger.info(
                "Skip user promotion inline train: train_n=%s < min_train_moves=%s",
                len(train),
                min_train_moves,
            )
            parent_metrics = _score_parent(resolved_parent, val)
            print(format_eval_metrics("parent", parent_metrics))
            print("skipped_train=true")
            print("primary=top1_sf_disagree")
            print("no_promote=true")
            return 0
        try:
            weights_path = _keras_parent_path(resolved_parent)
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            return 1
        trainer = _make_user_trainer(
            epochs=epochs,
            style_disagree_boost=style_disagree_boost,
            style_disagree_scale=style_disagree_scale,
            baseline_disagree_boost=baseline_disagree_boost,
            recency_boost=recency_boost,
        )
        logger.info(
            "Inline user finetune account=%s train_n=%s epochs=%s parent=%s",
            account_id,
            len(train),
            epochs,
            resolved_parent,
        )
        t0 = time.perf_counter()
        try:
            model, _train_metrics = trainer.fit(
                train,
                weights_path=weights_path,
                baseline_weights_path=weights_path,
                recency_lambda=recency_lambda,
                end_time_by_game_id=end_times,
                require_parent_weights=True,
            )
        except (RuntimeError, FileNotFoundError) as exc:
            logger.error("%s", exc)
            return 1
        logger.info("Inline user fit done in %.1fs", time.perf_counter() - t0)
        candidate_metrics = evaluate_datums(model, val)
        candidate_label = f"inline@{epochs}ep"
    else:
        assert candidate_uri is not None
        extra.append(f"candidate_uri={candidate_uri}")
        logger.info("Scoring candidate uri=%s on frozen user val n=%s", candidate_uri, len(val))
        candidate_metrics = evaluate_model_uri(candidate_uri, val)
        candidate_label = "candidate"

    parent_metrics = _score_parent(resolved_parent, val)
    _print_compare(
        candidate=candidate_metrics,
        parent=parent_metrics,
        candidate_label=candidate_label,
        parent_uri=resolved_parent,
        val_n=len(val),
        extra_lines=extra,
    )
    return 0


def run_offline_user_catch_up(
    *,
    account_id: str,
    parent_uri: str | None,
    max_rounds: int,
    batch_limit: int,
    min_new_moves: int,
    recency_lambda: float = DEFAULT_USER_RECENCY_LAMBDA,
    recency_boost: float = DEFAULT_USER_RECENCY_BOOST,
    style_disagree_boost: float = DEFAULT_USER_STYLE_DISAGREE_BOOST,
    style_disagree_scale: float = DEFAULT_USER_STYLE_DISAGREE_SCALE,
    baseline_disagree_boost: float = DEFAULT_USER_BASELINE_DISAGREE_BOOST,
    epochs: int = BaselineTrainer.DEFAULT_EPOCHS,
    min_train_moves: int = DEFAULT_MIN_USER_TRAIN_MOVES,
    limit: int | None = None,
    output_dir: str | Path | None = None,
    split_version: str = DEFAULT_SPLIT_SALT,
) -> int:
    if limit is not None:
        logger.warning(
            "Ignoring --limit=%s on user catch-up; personal queue uses --batch-limit.",
            limit,
        )
    db = get_db_client()
    store = TrainingDataStore(db)
    extra_where = account_id_in_sql(linked_account_ids_for_account(db, account_id))
    frozen_val = store.fetch_split_val_datums(
        split_version=split_version,
        extra_where=extra_where,
    )
    print(f"\npersonal queue (split_version={split_version})")
    print(f"val_moves={len(frozen_val)}")

    if len(frozen_val) < DEFAULT_MIN_USER_VAL_MOVES:
        logger.error("Val split too small: %s moves", len(frozen_val))
        return 1

    try:
        resolved_parent = resolve_user_parent_uri(db, parent_uri)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    try:
        baseline_path = _keras_parent_path(resolved_parent)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    trainer = _make_user_trainer(
        epochs=epochs,
        style_disagree_boost=style_disagree_boost,
        style_disagree_scale=style_disagree_scale,
        baseline_disagree_boost=baseline_disagree_boost,
        recency_boost=recency_boost,
    )
    registry = get_split_registry(db, split_version=split_version)
    max_rounds = max(1, int(max_rounds))
    curve: list[tuple[int, str, int, EvalMetrics]] = []

    out_ctx: Any
    if output_dir is not None:
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        out_ctx = nullcontext(str(out_root))
    else:
        out_ctx = tempfile.TemporaryDirectory(prefix="offline_user_catch_up_")

    with out_ctx as root:
        root_path = Path(root)
        resume_path = baseline_path
        round_i = 0
        last_unprocessed = 0
        while round_i < max_rounds:
            n_before = store.count_unprocessed_train(
                split_version=split_version,
                flag_column=PROCESSED_FLAG_PERSONAL,
                extra_where=extra_where,
            )
            last_unprocessed = n_before
            logger.info(
                "Offline user catch-up check round=%s unprocessed_train=%s min=%s batch_cap=%s",
                round_i + 1,
                n_before,
                min_new_moves,
                batch_limit,
            )
            if n_before < min_new_moves:
                logger.info(
                    "Caught up: unprocessed_train=%s < min=%s (remainder left for later).",
                    n_before,
                    min_new_moves,
                )
                _print_val_curve(curve)
                return 0

            datums, game_ids = store.fetch_unprocessed_train_batch(
                split_version=split_version,
                limit=batch_limit,
                flag_column=PROCESSED_FLAG_PERSONAL,
                extra_where=extra_where,
            )
            if not datums or not game_ids:
                logger.error(
                    "Pending=%s but unprocessed personal train batch empty.",
                    n_before,
                )
                _print_val_curve(curve)
                return 2

            round_i += 1
            last_id = game_ids[-1]
            logger.info(
                "=== offline user catch-up round %s/%s train_n=%s games=%s resume=%s last_game_id=%s ===",
                round_i,
                max_rounds,
                len(datums),
                len(game_ids),
                resume_path,
                last_id,
            )
            end_times = store.fetch_end_times(game_ids)
            try:
                model, _metrics = trainer.fit(
                    datums,
                    weights_path=resume_path,
                    baseline_weights_path=baseline_path,
                    recency_lambda=recency_lambda,
                    end_time_by_game_id=end_times,
                    require_parent_weights=True,
                )
            except (RuntimeError, FileNotFoundError) as exc:
                logger.error("%s", exc)
                _print_val_curve(curve)
                return 1
            save_path = root_path / f"round_{round_i}" / "model.keras"
            BaselineTrainer.save(model, save_path)
            val_metrics = evaluate_datums(model, frozen_val)
            marked = registry.mark_processed(game_ids, flag_column=PROCESSED_FLAG_PERSONAL)
            if marked <= 0:
                logger.error(
                    "Fit succeeded but mark_processed updated 0 rows (games=%s) - "
                    "stop to avoid infinite loop.",
                    len(game_ids),
                )
                _print_val_curve(curve)
                return 2
            resume_path = save_path
            curve.append((round_i, last_id, len(datums), val_metrics))
            print(format_eval_metrics(f"round{round_i}", val_metrics))

            n_after = store.count_unprocessed_train(
                split_version=split_version,
                flag_column=PROCESSED_FLAG_PERSONAL,
                extra_where=extra_where,
            )
            last_unprocessed = n_after
            if n_after >= n_before:
                logger.error(
                    "Train succeeded but unprocessed count did not drop "
                    "(before=%s after=%s) - stop to avoid infinite loop.",
                    n_before,
                    n_after,
                )
                _print_val_curve(curve)
                return 2

        if last_unprocessed >= min_new_moves:
            logger.error(
                "Hit max_rounds=%s with unprocessed still above min - stopping.",
                max_rounds,
            )
            _print_val_curve(curve)
            return 3
        _print_val_curve(curve)
        return 0


def _add_common_user_args(
    parser: argparse.ArgumentParser,
    *,
    parent_flags: tuple[str, ...] = ("--parent-uri",),
) -> None:
    parser.add_argument("--account-id", type=str, required=True)
    parser.add_argument(
        *parent_flags,
        dest="parent_uri",
        type=str,
        default=None,
        help="Parent Keras / MLflow URI (default: latest production baseline).",
    )
    parser.add_argument(
        "--recency-lambda",
        type=float,
        default=DEFAULT_USER_RECENCY_LAMBDA,
    )
    parser.add_argument(
        "--recency-boost",
        type=float,
        default=DEFAULT_USER_RECENCY_BOOST,
    )
    parser.add_argument(
        "--style-disagree-boost",
        type=float,
        default=DEFAULT_USER_STYLE_DISAGREE_BOOST,
    )
    parser.add_argument(
        "--style-disagree-scale",
        type=float,
        default=DEFAULT_USER_STYLE_DISAGREE_SCALE,
    )
    parser.add_argument(
        "--baseline-disagree-boost",
        type=float,
        default=DEFAULT_USER_BASELINE_DISAGREE_BOOST,
    )
    parser.add_argument("--epochs", type=int, default=BaselineTrainer.DEFAULT_EPOCHS)
    parser.add_argument(
        "--min-train-moves",
        type=int,
        default=DEFAULT_MIN_USER_TRAIN_MOVES,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional cap on TRAIN game ids before hydrate "
            "(sorted game_id, complete games; val/test are never truncated). "
            "Ignored by catch-up (uses --batch-limit)."
        ),
    )


def build_finetune_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline user finetune + val eval (no DB write).")
    _add_common_user_args(parser)
    return parser


def build_promotion_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline user promotion compare on frozen hash-split val (no DB write)."
    )
    _add_common_user_args(parser, parent_flags=("--parent-uri", "--baseline-uri"))
    parser.add_argument("--candidate-uri", type=str, default=None)
    parser.add_argument(
        "--train-inline",
        action="store_true",
        help="Finetune from parent on frozen user train, then compare on val.",
    )
    return parser


def build_catch_up_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline user catch-up on personal registry train queue "
            "(marks personal flags; no promote / baseline writes)."
        )
    )
    _add_common_user_args(parser)
    parser.add_argument("--max-rounds", type=int, default=50)
    parser.add_argument("--min-new-moves", type=int, default=DEFAULT_MIN_USER_NEW_MOVES)
    parser.add_argument("--batch-limit", type=int, default=DEFAULT_USER_BATCH_LIMIT)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser


def main_finetune_eval() -> int:
    args = build_finetune_eval_parser().parse_args()
    return run_offline_user_finetune_eval(
        account_id=str(args.account_id),
        parent_uri=str(args.parent_uri) if args.parent_uri else None,
        recency_lambda=float(args.recency_lambda),
        recency_boost=float(args.recency_boost),
        style_disagree_boost=float(args.style_disagree_boost),
        style_disagree_scale=float(args.style_disagree_scale),
        baseline_disagree_boost=float(args.baseline_disagree_boost),
        epochs=max(1, int(args.epochs)),
        min_train_moves=max(1, int(args.min_train_moves)),
        limit=int(args.limit) if args.limit is not None else None,
    )


def main_promotion() -> int:
    args = build_promotion_parser().parse_args()
    return run_offline_user_promotion(
        account_id=str(args.account_id),
        candidate_uri=str(args.candidate_uri) if args.candidate_uri else None,
        train_inline=bool(args.train_inline),
        parent_uri=str(args.parent_uri) if args.parent_uri else None,
        recency_lambda=float(args.recency_lambda),
        recency_boost=float(args.recency_boost),
        style_disagree_boost=float(args.style_disagree_boost),
        style_disagree_scale=float(args.style_disagree_scale),
        baseline_disagree_boost=float(args.baseline_disagree_boost),
        epochs=max(1, int(args.epochs)),
        min_train_moves=max(1, int(args.min_train_moves)),
        limit=int(args.limit) if args.limit is not None else None,
    )


def main_catch_up() -> int:
    args = build_catch_up_parser().parse_args()
    return run_offline_user_catch_up(
        account_id=str(args.account_id),
        parent_uri=str(args.parent_uri) if args.parent_uri else None,
        max_rounds=max(1, int(args.max_rounds)),
        batch_limit=max(1, int(args.batch_limit)),
        min_new_moves=max(1, int(args.min_new_moves)),
        recency_lambda=float(args.recency_lambda),
        recency_boost=float(args.recency_boost),
        style_disagree_boost=float(args.style_disagree_boost),
        style_disagree_scale=float(args.style_disagree_scale),
        baseline_disagree_boost=float(args.baseline_disagree_boost),
        epochs=max(1, int(args.epochs)),
        min_train_moves=max(1, int(args.min_train_moves)),
        limit=int(args.limit) if args.limit is not None else None,
        output_dir=str(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    from chess_teacher.utils.process_utils import run_script_main

    run_script_main(main_finetune_eval)
