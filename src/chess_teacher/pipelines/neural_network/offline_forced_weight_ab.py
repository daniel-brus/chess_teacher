"""Offline E21 A/B: forced-move downweight on/off (registry queue).

One change family: ``forced_scale_pawns`` only. Same style-disagree knobs,
epochs, batch, and train games for both arms. Val packed once.

Control: forced off. Treatment: continuous forced downweight (default scale 1.5).

``--encoder mlp`` = flat BaselineTrainer (legacy). ``--encoder hybrid`` =
HybridBoardTrainer (preferred continuum).
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal

from chess_teacher.pipelines.neural_network.board_encoder import HybridBoardTrainer
from chess_teacher.pipelines.neural_network.create_training_set import TrainingDataStore
from chess_teacher.pipelines.neural_network.eval_metrics import (
    EvalMetrics,
    evaluate_packed,
    format_eval_delta,
    format_eval_metrics,
    pack_datums_for_eval,
)
from chess_teacher.pipelines.neural_network.offline_eval import load_registry_val_datums
from chess_teacher.pipelines.neural_network.pipeline_steps import (
    MAX_MOVES_PER_BASELINE_BATCH,
    MIN_NEW_MOVES_BASELINE,
)
from chess_teacher.pipelines.neural_network.ply_weights import DEFAULT_FORCED_SCALE_PAWNS
from chess_teacher.pipelines.neural_network.split_registry import (
    PROCESSED_FLAG_BASELINE,
    get_split_registry,
)
from chess_teacher.pipelines.neural_network.splits import DEFAULT_SPLIT_SALT
from chess_teacher.pipelines.neural_network.train import BaselineTrainer
from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.logging import get_logger

logger = get_logger()

EncoderKind = Literal["mlp", "hybrid"]


def _append_curve_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _make_trainers(
    *,
    encoder: EncoderKind,
    epochs: int,
    style_disagree_boost: float,
    style_disagree_scale: float,
    forced_scale_pawns: float,
) -> tuple[Any, Any, Any]:
    """Return (control, treatment, save_fn)."""
    if encoder == "hybrid":
        return (
            HybridBoardTrainer(
                epochs=epochs,
                style_disagree_boost=style_disagree_boost,
                style_disagree_scale=style_disagree_scale,
                forced_scale_pawns=None,
            ),
            HybridBoardTrainer(
                epochs=epochs,
                style_disagree_boost=style_disagree_boost,
                style_disagree_scale=style_disagree_scale,
                forced_scale_pawns=forced_scale_pawns,
            ),
            HybridBoardTrainer.save,
        )
    return (
        BaselineTrainer(
            epochs=epochs,
            style_disagree_boost=style_disagree_boost,
            style_disagree_scale=style_disagree_scale,
            forced_scale_pawns=None,
        ),
        BaselineTrainer(
            epochs=epochs,
            style_disagree_boost=style_disagree_boost,
            style_disagree_scale=style_disagree_scale,
            forced_scale_pawns=forced_scale_pawns,
        ),
        BaselineTrainer.save,
    )


def run_forced_weight_ab(
    *,
    split_version: str,
    full_val: bool,
    val_limit: int,
    max_rounds: int,
    min_new_moves: int,
    batch_limit: int,
    epochs: int,
    style_disagree_boost: float,
    style_disagree_scale: float,
    forced_scale_pawns: float,
    output_dir: str | Path | None,
    reset_queue: bool,
    encoder: EncoderKind = "mlp",
    start_round: int = 1,
    parent_off: str | Path | None = None,
    parent_on: str | Path | None = None,
) -> int:
    db = get_db_client()
    store = TrainingDataStore(db)
    registry = get_split_registry(db, split_version=split_version)

    if reset_queue:
        cleared = registry.clear_processed(flag_column=PROCESSED_FLAG_BASELINE)
        logger.warning(
            "Reset baseline train queue: clear_processed updated=%s",
            cleared,
        )

    val = load_registry_val_datums(
        db,
        split_version=split_version,
        limit=None if full_val else val_limit,
        full=full_val,
    )
    if len(val) < 10:
        logger.error("Val set too small: %s", len(val))
        return 1
    packed_val = pack_datums_for_eval(val)
    print(
        f"\npacked val n_input={packed_val.n_input} kept={len(packed_val.kept_datums)} "
        f"encoder={encoder} forced_scale={forced_scale_pawns}"
    )

    control, treatment, save_fn = _make_trainers(
        encoder=encoder,
        epochs=epochs,
        style_disagree_boost=style_disagree_boost,
        style_disagree_scale=style_disagree_scale,
        forced_scale_pawns=forced_scale_pawns,
    )

    out_ctx: Any
    if output_dir is not None:
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        out_ctx = nullcontext(str(out_root))
    else:
        out_ctx = tempfile.TemporaryDirectory(prefix="offline_forced_ab_")

    max_rounds = max(1, int(max_rounds))
    start_round = max(1, int(start_round))
    end_round = start_round + max_rounds - 1
    curve: list[tuple[int, str, int, EvalMetrics, EvalMetrics]] = []
    seed_off = Path(parent_off) if parent_off else None
    seed_on = Path(parent_on) if parent_on else None
    if seed_off is not None and not seed_off.is_file():
        logger.error("parent-off missing: %s", seed_off)
        return 1
    if seed_on is not None and not seed_on.is_file():
        logger.error("parent-on missing: %s", seed_on)
        return 1
    if (seed_off is None) ^ (seed_on is None):
        logger.error("Provide both --parent-off and --parent-on, or neither")
        return 1

    with out_ctx as root:
        root_path = Path(root)
        cur_parent_off: Path | None = seed_off
        cur_parent_on: Path | None = seed_on
        round_i = start_round - 1
        if seed_off is not None:
            logger.info(
                "Continuing forced A/B from parents off=%s on=%s start_round=%s end_round=%s",
                seed_off,
                seed_on,
                start_round,
                end_round,
            )

        while round_i < end_round:
            n_before = store.count_unprocessed_train(split_version=split_version)
            logger.info(
                "Forced-weight A/B check encoder=%s round=%s unprocessed=%s min=%s",
                encoder,
                round_i + 1,
                n_before,
                min_new_moves,
            )
            if n_before < min_new_moves:
                logger.info("Caught up early unprocessed=%s", n_before)
                break

            datums, game_ids = store.fetch_unprocessed_train_batch(
                split_version=split_version,
                limit=batch_limit,
            )
            if not datums or not game_ids:
                logger.error("Empty unprocessed batch with pending=%s", n_before)
                return 2

            round_i += 1
            last_id = game_ids[-1]
            print(
                f"\n=== round {round_i}/{end_round} encoder={encoder} "
                f"train_n={len(datums)} games={len(game_ids)} last_game_id={last_id} ==="
            )
            t0 = time.monotonic()

            model_off, _ = control.fit(datums, weights_path=cur_parent_off)
            m_off = evaluate_packed(model_off, packed_val)
            save_off = root_path / f"round_{round_i}" / "forced_off.keras"
            save_fn(model_off, save_off)
            cur_parent_off = save_off
            print(format_eval_metrics(f"forced_off_r{round_i}", m_off))

            model_on, _ = treatment.fit(datums, weights_path=cur_parent_on)
            m_on = evaluate_packed(model_on, packed_val)
            save_on = root_path / f"round_{round_i}" / "forced_on.keras"
            save_fn(model_on, save_on)
            cur_parent_on = save_on
            print(format_eval_metrics(f"forced_on_r{round_i}", m_on))
            print(
                format_eval_delta(
                    m_on,
                    m_off,
                    candidate_name=f"forced_on_r{round_i}",
                    baseline_name=f"forced_off_r{round_i}",
                )
            )
            d_on = m_on.top1_sf_disagree
            d_off = m_off.top1_sf_disagree
            a_on = m_on.top1_sf_agree
            a_off = m_off.top1_sf_agree
            if d_on is not None and d_off is not None:
                print(
                    f"primary delta disagree_t1 (on - off) round={round_i} = "
                    f"{float(d_on) - float(d_off):+.4f}"
                )
            if a_on is not None and a_off is not None:
                print(
                    f"guardrail delta agree_t1 (on - off) round={round_i} = "
                    f"{float(a_on) - float(a_off):+.4f}"
                )

            marked = registry.mark_processed(game_ids)
            if marked <= 0:
                logger.error("mark_processed updated 0 rows")
                return 2
            n_after = store.count_unprocessed_train(split_version=split_version)
            curve.append((round_i, last_id, len(datums), m_off, m_on))
            _append_curve_jsonl(
                root_path / "val_curve.jsonl",
                {
                    "round": round_i,
                    "encoder": encoder,
                    "last_game_id": last_id,
                    "train_n": len(datums),
                    "games": len(game_ids),
                    "marked": marked,
                    "pending_before": n_before,
                    "pending_after": n_after,
                    "elapsed_s": round(time.monotonic() - t0, 1),
                    "forced_scale_pawns": forced_scale_pawns,
                    "forced_off": m_off.as_dict(),
                    "forced_on": m_on.as_dict(),
                },
            )
            if n_after >= n_before:
                logger.error("Unprocessed did not drop before=%s after=%s", n_before, n_after)
                return 2

    print(
        f"\n=== forced-weight A/B curve encoder={encoder} "
        f"(change family = forced scale only) ==="
    )
    for r_i, last_id, n_train, m_off, m_on in curve:
        print(f"round={r_i} last_game_id={last_id} train_n={n_train}")
        print(format_eval_metrics(f"  forced_off_r{r_i}", m_off))
        print(format_eval_metrics(f"  forced_on_r{r_i}", m_on))
        if m_on.top1_sf_disagree is not None and m_off.top1_sf_disagree is not None:
            print(
                f"  delta disagree_t1 (on-off)="
                f"{float(m_on.top1_sf_disagree) - float(m_off.top1_sf_disagree):+.4f}"
            )
    if not curve:
        return 1
    return 0 if len(curve) >= max_rounds else 3


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-version", type=str, default=DEFAULT_SPLIT_SALT)
    p.add_argument("--limit", type=int, default=10000, help="Val sample size if not --full-val")
    p.add_argument("--full-val", action="store_true")
    p.add_argument("--max-rounds", type=int, default=3)
    p.add_argument("--min-new-moves", type=int, default=MIN_NEW_MOVES_BASELINE)
    p.add_argument("--batch-limit", type=int, default=MAX_MOVES_PER_BASELINE_BATCH)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--style-disagree-boost", type=float, default=2.0)
    p.add_argument("--style-disagree-scale", type=float, default=2.0)
    p.add_argument(
        "--forced-scale-pawns",
        type=float,
        default=DEFAULT_FORCED_SCALE_PAWNS,
        help="Treatment arm forced downweight scale (control stays off).",
    )
    p.add_argument(
        "--encoder",
        type=str,
        choices=("mlp", "hybrid"),
        default="mlp",
        help="Trainer family for both arms (default mlp for back-compat).",
    )
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--reset-queue", action="store_true")
    p.add_argument(
        "--start-round",
        type=int,
        default=1,
        help="First round index to write (use >1 to continue without overwriting).",
    )
    p.add_argument(
        "--parent-off",
        type=str,
        default=None,
        help="Seed control arm weights (pair with --parent-on).",
    )
    p.add_argument(
        "--parent-on",
        type=str,
        default=None,
        help="Seed treatment arm weights (pair with --parent-off).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return run_forced_weight_ab(
        split_version=str(args.split_version),
        full_val=bool(args.full_val),
        val_limit=max(50, int(args.limit)),
        max_rounds=max(1, int(args.max_rounds)),
        min_new_moves=max(1, int(args.min_new_moves)),
        batch_limit=max(1, int(args.batch_limit)),
        epochs=max(1, int(args.epochs)),
        style_disagree_boost=float(args.style_disagree_boost),
        style_disagree_scale=float(args.style_disagree_scale),
        forced_scale_pawns=float(args.forced_scale_pawns),
        output_dir=str(args.output_dir) if args.output_dir else None,
        reset_queue=bool(args.reset_queue),
        encoder=str(args.encoder),  # type: ignore[arg-type]
        start_round=max(1, int(args.start_round)),
        parent_off=str(args.parent_off) if args.parent_off else None,
        parent_on=str(args.parent_on) if args.parent_on else None,
    )
