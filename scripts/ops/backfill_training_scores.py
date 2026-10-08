"""Score registered models that have no ``training_scores`` row yet.

Uses the same registry-val prefix as training (platform val for baselines,
that user's val for personal models). Points are dated at ``trained_at``.
Models that already have a score row are left alone. Incompatible heads and
missing weights are skipped.

Local::

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/ops/backfill_training_scores.py --dry-run

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/ops/backfill_training_scores.py

k3d / prod::

    python scripts/utils/run_script_job.py backfill_training_scores -- --dry-run
    python scripts/utils/run_script_job.py backfill_training_scores -- --follow
"""

from __future__ import annotations

import argparse
import gc
from typing import Any

from chess_teacher.pipelines.neural_network.create_training_set import account_id_in_sql
from chess_teacher.pipelines.neural_network.models import BaselineModel, PersonalModel
from chess_teacher.pipelines.neural_network.offline_eval import load_registry_val_datums
from chess_teacher.pipelines.neural_network.pipeline_steps import MAX_MOVES_PER_REGISTRY_VAL_EVAL
from chess_teacher.pipelines.neural_network.splits import DEFAULT_SPLIT_SALT
from chess_teacher.pipelines.neural_network.training_score_backfill import (
    ScoreTarget,
    apply_limit,
    group_targets,
    iter_batches,
    select_score_targets,
    training_score_for_backfill,
)
from chess_teacher.pipelines.neural_network.training_scores import TrainingScore
from chess_teacher.platform.user import User
from chess_teacher.utils.db.client import DatabaseClient, get_db_client
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.process_utils import log_script_runtime_context, run_script_main

logger = get_logger()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List models that would be scored. Does not load weights or write rows.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Score at most this many pending models (oldest first).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="How many models to hold in memory while scoring one exam. Default 1.",
    )
    parser.add_argument(
        "--split-version",
        default=DEFAULT_SPLIT_SALT,
        help=f"Registry split label. Default {DEFAULT_SPLIT_SALT}.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        logger.error("--limit must be at least 1")
        return 2
    if args.batch_size < 1:
        logger.error("--batch-size must be at least 1")
        return 2

    log_script_runtime_context(logger, script="backfill_training_scores")
    db_client = get_db_client()
    targets = apply_limit(_pending_targets(db_client), args.limit)
    _print_pending(targets)
    if args.dry_run:
        print("dry_run=true")
        return 0
    if not targets:
        print("scored=0 failed=0")
        return 0

    scored = 0
    failed = 0
    for user_id, group in group_targets(targets):
        written, group_failed = _score_group(
            db_client,
            user_id,
            group,
            batch_size=args.batch_size,
            split_version=args.split_version,
        )
        scored += written
        failed += group_failed
    print(f"scored={scored} failed={failed}")
    logger.info("Training score backfill scored=%s failed=%s", scored, failed)
    return 1 if failed else 0


def _pending_targets(db_client: DatabaseClient) -> list[ScoreTarget]:
    baselines = BaselineModel.fetch_all_from_db(db_client)
    personals = PersonalModel.fetch_all_from_db(db_client)
    scored_ids = {row.model_id for row in TrainingScore.fetch_all_from_db(db_client)}
    return select_score_targets(baselines, personals, scored_ids)


def _print_pending(targets: list[ScoreTarget]) -> None:
    for target in targets:
        who = "baseline" if target.user_id is None else f"user={target.user_id}"
        print(
            f"pending {who} version={target.version} model_id={target.model_id} "
            f"trained_at={target.trained_at.isoformat()}"
        )
    print(f"pending_count={len(targets)}")


def _score_group(
    db_client: DatabaseClient,
    user_id: str | None,
    targets: list[ScoreTarget],
    *,
    batch_size: int,
    split_version: str,
) -> tuple[int, int]:
    extra_where = _exam_where(db_client, user_id)
    if extra_where == "1 = 0":
        logger.warning(
            "No linked accounts for user=%s; skipping %s models",
            user_id,
            len(targets),
        )
        return 0, len(targets)

    datums = load_registry_val_datums(
        db_client,
        split_version=split_version,
        limit=MAX_MOVES_PER_REGISTRY_VAL_EVAL,
        extra_where=extra_where,
    )
    if not datums:
        logger.warning("Empty val set user=%s; skipping %s models", user_id, len(targets))
        return 0, len(targets)

    from chess_teacher.pipelines.neural_network.eval_metrics import (
        EvalMetrics,
        score_models_on_datums,
    )
    from chess_teacher.pipelines.neural_network.train import (
        clear_candidate_style_model_cache,
        load_candidate_style_from_uri,
    )

    written = 0
    failed = 0
    for batch in iter_batches(targets, batch_size):
        loaded: dict[str, Any] = {}
        for target in batch:
            try:
                loaded[target.model_id] = load_candidate_style_from_uri(target.model_uri)
            except Exception:
                logger.exception(
                    "Failed to load model_id=%s version=%s",
                    target.model_id,
                    target.version,
                )
                failed += 1
        scored: dict[str, EvalMetrics] | None = None
        if loaded:
            try:
                scored = score_models_on_datums(loaded, datums)
            except Exception:
                logger.exception("Failed to score batch user=%s n=%s", user_id, len(loaded))
                failed += len(loaded)
            if scored is not None:
                for target in batch:
                    if target.model_id not in loaded:
                        continue
                    metrics = scored.get(target.model_id)
                    if metrics is None:
                        failed += 1
                        continue
                    training_score_for_backfill(target, metrics).save_to_db(db_client)
                    written += 1
                    print(
                        f"scored model_id={target.model_id} version={target.version} "
                        f"val_top1={metrics.top1_overall:.4f} "
                        f"sf_delta_mean_pawns={metrics.sf_delta_mean_pawns:.4f}"
                    )
        loaded.clear()
        clear_candidate_style_model_cache()
        gc.collect()
    return written, failed


def _exam_where(db_client: DatabaseClient, user_id: str | None) -> str | None:
    if user_id is None:
        return None
    try:
        user = User.fetch_from_db(db_client, id=user_id)
    except Exception:
        logger.exception("Skipping user=%s; user row missing", user_id)
        return "1 = 0"
    account_ids = [account.account_id for account in user.get_linked_accounts(db_client)]
    return account_id_in_sql(account_ids)


if __name__ == "__main__":
    run_script_main(main)
