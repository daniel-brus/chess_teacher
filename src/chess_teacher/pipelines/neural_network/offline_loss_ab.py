"""Offline E22 A/B: candidate loss variants on registry queue (no prod wiring).

One change family: ``loss_kind`` only (``sparse`` | ``soft`` | ``sf_mix``).
Same hybrid/MLP encoder, style knobs, epochs, shared train batches. Val packed once.

Also reports le-6-piece and kings+pawns agree/disagree (Play-floor slices).

Example::

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/tools/offline_loss_ab.py ^
      --encoder hybrid --epochs 10 --max-rounds 3 --limit 10000 --reset-queue ^
      --loss-kinds sparse,soft,sf_mix ^
      --output-dir storage/tmp/hybrid_loss_ab_r3_ep10
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal

import chess
import numpy as np

from chess_teacher.pipelines.neural_network.board_encoder import HybridBoardTrainer
from chess_teacher.pipelines.neural_network.candidate_losses import (
    DEFAULT_SF_MIX_ALPHA,
    DEFAULT_SOFT_TEMPERATURE_PAWNS,
    LossKind,
)
from chess_teacher.pipelines.neural_network.create_training_set import TrainingDataStore
from chess_teacher.pipelines.neural_network.eval_metrics import (
    EvalMetrics,
    compute_candidate_style_metrics,
    evaluate_packed,
    format_eval_metrics,
    pack_datums_for_eval,
    predict_candidate_logits,
)
from chess_teacher.pipelines.neural_network.material_regime import only_kings_and_pawns
from chess_teacher.pipelines.neural_network.offline_eval import load_registry_val_datums
from chess_teacher.pipelines.neural_network.pipeline_steps import (
    MAX_MOVES_PER_BASELINE_BATCH,
    MIN_NEW_MOVES_BASELINE,
)
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


def _metrics_on_selector(
    *,
    logits: np.ndarray,
    packed: Any,
    selector: np.ndarray,
) -> EvalMetrics | None:
    idx = np.flatnonzero(selector)
    if idx.size < 5:
        return None
    return compute_candidate_style_metrics(
        logits=logits[idx],
        mask=packed.mask[idx],
        labels=packed.labels[idx],
        move_feats=packed.feats[idx],
        plies=[packed.kept_datums[i].ply for i in idx],
        n_input=int(idx.size),
        max_candidates=packed.max_candidates,
    )


def _slice_selectors(packed: Any) -> dict[str, np.ndarray]:
    counts = [len(chess.Board(d.fen_before).piece_map()) for d in packed.kept_datums]
    kap = [only_kings_and_pawns(d.fen_before) for d in packed.kept_datums]
    n = len(packed.kept_datums)
    return {
        "all": np.ones(n, dtype=bool),
        "pieces_le_6": np.asarray([c <= 6 for c in counts], dtype=bool),
        "kings_pawns": np.asarray(kap, dtype=bool),
    }


def _metrics_bundle(model: Any, packed: Any) -> dict[str, Any]:
    logits = predict_candidate_logits(model, packed.kept_datums, packed.feats)
    out: dict[str, Any] = {}
    for name, sel in _slice_selectors(packed).items():
        m = _metrics_on_selector(logits=logits, packed=packed, selector=sel)
        out[name] = None if m is None else m.as_dict()
    # Keep primary overall via evaluate_packed for consistency with other A/Bs.
    overall = evaluate_packed(model, packed)
    out["all"] = overall.as_dict()
    return out


def _make_trainer(
    *,
    encoder: EncoderKind,
    epochs: int,
    style_disagree_boost: float,
    style_disagree_scale: float,
    loss_kind: LossKind,
    soft_temperature_pawns: float,
    sf_mix_alpha: float,
) -> Any:
    kwargs: dict[str, Any] = {
        "epochs": epochs,
        "style_disagree_boost": style_disagree_boost,
        "style_disagree_scale": style_disagree_scale,
        "loss_kind": loss_kind,
        "soft_temperature_pawns": soft_temperature_pawns,
        "sf_mix_alpha": sf_mix_alpha,
    }
    if encoder == "hybrid":
        return HybridBoardTrainer(**kwargs)
    return BaselineTrainer(**kwargs)


def _save_fn(encoder: EncoderKind):
    return HybridBoardTrainer.save if encoder == "hybrid" else BaselineTrainer.save


def run_loss_ab(
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
    loss_kinds: list[LossKind],
    soft_temperature_pawns: float,
    sf_mix_alpha: float,
    output_dir: str | Path | None,
    reset_queue: bool,
    encoder: EncoderKind = "hybrid",
) -> int:
    if not loss_kinds:
        logger.error("Need at least one --loss-kinds entry")
        return 1
    for kind in loss_kinds:
        if kind not in ("sparse", "soft", "sf_mix"):
            logger.error("Unknown loss_kind=%s", kind)
            return 1

    db = get_db_client()
    store = TrainingDataStore(db)
    registry = get_split_registry(db, split_version=split_version)

    if reset_queue:
        cleared = registry.clear_processed(flag_column=PROCESSED_FLAG_BASELINE)
        logger.warning("Reset baseline train queue: clear_processed updated=%s", cleared)

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
        f"encoder={encoder} loss_kinds={loss_kinds}"
    )

    trainers = {
        kind: _make_trainer(
            encoder=encoder,
            epochs=epochs,
            style_disagree_boost=style_disagree_boost,
            style_disagree_scale=style_disagree_scale,
            loss_kind=kind,
            soft_temperature_pawns=soft_temperature_pawns,
            sf_mix_alpha=sf_mix_alpha,
        )
        for kind in loss_kinds
    }
    save_fn = _save_fn(encoder)

    out_ctx: Any
    if output_dir is not None:
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        out_ctx = nullcontext(str(out_root))
    else:
        out_ctx = tempfile.TemporaryDirectory(prefix="offline_loss_ab_")

    max_rounds = max(1, int(max_rounds))
    parents: dict[str, Path | None] = {kind: None for kind in loss_kinds}

    with out_ctx as root:
        root_path = Path(root)
        for round_i in range(1, max_rounds + 1):
            n_before = store.count_unprocessed_train(split_version=split_version)
            logger.info(
                "Loss A/B check encoder=%s round=%s unprocessed=%s kinds=%s",
                encoder,
                round_i,
                n_before,
                loss_kinds,
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

            last_id = game_ids[-1]
            print(
                f"\n=== round {round_i}/{max_rounds} encoder={encoder} "
                f"train_n={len(datums)} games={len(game_ids)} last_game_id={last_id} ==="
            )
            t0 = time.monotonic()
            arm_metrics: dict[str, dict[str, Any]] = {}

            for kind in loss_kinds:
                trainer = trainers[kind]
                model, _ = trainer.fit(datums, weights_path=parents[kind])
                bundle = _metrics_bundle(model, packed_val)
                arm_metrics[kind] = bundle
                save_path = root_path / f"round_{round_i}" / f"{kind}.keras"
                save_fn(model, save_path)
                parents[kind] = save_path
                overall = bundle["all"]
                print(
                    format_eval_metrics(
                        f"{kind}_r{round_i}",
                        EvalMetrics(
                            top1_overall=float(overall["top1_overall"]),
                            top3_overall=float(overall["top3_overall"]),
                            top1_overall_weighted=float(overall["top1_overall_weighted"]),
                            n_eval=float(overall["n_eval"]),
                            n_dropped=float(overall["n_dropped"]),
                            n_sf_agree=float(overall["n_sf_agree"]),
                            n_sf_disagree=float(overall["n_sf_disagree"]),
                            sf_disagree_frac=float(overall["sf_disagree_frac"]),
                            top1_sf_agree=overall.get("top1_sf_agree"),
                            top3_sf_agree=overall.get("top3_sf_agree"),
                            top1_sf_disagree=overall.get("top1_sf_disagree"),
                            top3_sf_disagree=overall.get("top3_sf_disagree"),
                        ),
                    )
                )
                for slice_name in ("pieces_le_6", "kings_pawns"):
                    sl = bundle.get(slice_name)
                    if not sl:
                        print(f"  {kind} {slice_name}: (too few)")
                        continue
                    print(
                        f"  {kind} {slice_name}: disagree_t1={sl.get('top1_sf_disagree')} "
                        f"agree_t1={sl.get('top1_sf_agree')} n={sl.get('n_eval')}"
                    )

            control = "sparse" if "sparse" in arm_metrics else loss_kinds[0]
            if control in arm_metrics:
                for kind in loss_kinds:
                    if kind == control:
                        continue
                    c = arm_metrics[control]["all"]
                    t = arm_metrics[kind]["all"]
                    if (
                        c.get("top1_sf_disagree") is not None
                        and t.get("top1_sf_disagree") is not None
                    ):
                        print(
                            f"  delta disagree_t1 ({kind}-{control})="
                            f"{float(t['top1_sf_disagree']) - float(c['top1_sf_disagree']):+.4f}"
                        )
                    if c.get("top1_sf_agree") is not None and t.get("top1_sf_agree") is not None:
                        print(
                            f"  delta agree_t1 ({kind}-{control})="
                            f"{float(t['top1_sf_agree']) - float(c['top1_sf_agree']):+.4f}"
                        )

            marked = registry.mark_processed(game_ids)
            if marked <= 0:
                logger.error("mark_processed updated 0 rows")
                return 2
            n_after = store.count_unprocessed_train(split_version=split_version)
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
                    "soft_temperature_pawns": soft_temperature_pawns,
                    "sf_mix_alpha": sf_mix_alpha,
                    "arms": arm_metrics,
                },
            )
            if n_after >= n_before:
                logger.error("Unprocessed did not drop before=%s after=%s", n_before, n_after)
                return 2

    print(f"\n=== loss A/B done encoder={encoder} kinds={loss_kinds} ===")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-version", type=str, default=DEFAULT_SPLIT_SALT)
    p.add_argument("--limit", type=int, default=10000)
    p.add_argument("--full-val", action="store_true")
    p.add_argument("--max-rounds", type=int, default=3)
    p.add_argument("--min-new-moves", type=int, default=MIN_NEW_MOVES_BASELINE)
    p.add_argument("--batch-limit", type=int, default=MAX_MOVES_PER_BASELINE_BATCH)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--style-disagree-boost", type=float, default=2.0)
    p.add_argument("--style-disagree-scale", type=float, default=2.0)
    p.add_argument(
        "--loss-kinds",
        type=str,
        default="sparse,soft,sf_mix",
        help="Comma list: sparse,soft,sf_mix",
    )
    p.add_argument("--soft-temperature-pawns", type=float, default=DEFAULT_SOFT_TEMPERATURE_PAWNS)
    p.add_argument("--sf-mix-alpha", type=float, default=DEFAULT_SF_MIX_ALPHA)
    p.add_argument(
        "--encoder",
        type=str,
        choices=("mlp", "hybrid"),
        default="hybrid",
    )
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--reset-queue", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    kinds = [
        k.strip()  # type: ignore[misc]
        for k in str(args.loss_kinds).split(",")
        if k.strip()
    ]
    return run_loss_ab(
        split_version=str(args.split_version),
        full_val=bool(args.full_val),
        val_limit=max(50, int(args.limit)),
        max_rounds=max(1, int(args.max_rounds)),
        min_new_moves=max(1, int(args.min_new_moves)),
        batch_limit=max(1, int(args.batch_limit)),
        epochs=max(1, int(args.epochs)),
        style_disagree_boost=float(args.style_disagree_boost),
        style_disagree_scale=float(args.style_disagree_scale),
        loss_kinds=kinds,  # type: ignore[arg-type]
        soft_temperature_pawns=float(args.soft_temperature_pawns),
        sf_mix_alpha=float(args.sf_mix_alpha),
        output_dir=str(args.output_dir) if args.output_dir else None,
        reset_queue=bool(args.reset_queue),
        encoder=str(args.encoder),  # type: ignore[arg-type]
    )
