"""Compare hybrid checkpoints on phase + sparse-material val slices.

Packs registry val once, scores each model once, then reports stratified
metrics on: all / opening / middle / endgame / pieces<=10 / pieces<=6 /
kings+pawns-only.

Does not train. Local keras paths OK as ``--model`` (repeatable).

Run::

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/tools/analyze_val_sparse_material.py ^
      --model R10=storage/tmp/hybrid_forced_ab_r10_ep10_s1.5/round_10/forced_on.keras ^
      --model R20=storage/tmp/hybrid_forced_ab_r10_ep10_s1.5/round_20/forced_on.keras ^
      --limit 10000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import chess
import numpy as np

from chess_teacher.pipelines.neural_network.create_training_set import TrainingDatum
from chess_teacher.pipelines.neural_network.eval_metrics import (
    EvalMetrics,
    compute_candidate_style_metrics,
    format_eval_metrics,
    pack_datums_for_eval,
    phase_from_features,
    predict_candidate_logits,
)
from chess_teacher.pipelines.neural_network.offline_eval import load_registry_val_datums
from chess_teacher.pipelines.neural_network.ply_weights import user_not_sf_best_mask
from chess_teacher.pipelines.neural_network.splits import DEFAULT_SPLIT_SALT
from chess_teacher.pipelines.neural_network.tf_runtime import ensure_tensorflow_logging
from chess_teacher.pipelines.neural_network.train import load_candidate_style_from_uri
from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.process_utils import log_script_runtime_context, run_script_main

ensure_tensorflow_logging()
logger = get_logger()


def _piece_count(fen: str) -> int:
    return len(chess.Board(fen).piece_map())


def _is_kings_and_pawns(fen: str) -> bool:
    board = chess.Board(fen)
    for piece in board.piece_map().values():
        if piece.piece_type not in (chess.KING, chess.PAWN):
            return False
    return True


def _parse_model_arg(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"Expected LABEL=path_or_uri, got {raw!r}")
    label, uri = raw.split("=", 1)
    label = label.strip()
    uri = uri.strip()
    if not label or not uri:
        raise argparse.ArgumentTypeError(f"Bad --model {raw!r}")
    return label, uri


def _metrics_on_selector(
    *,
    logits: np.ndarray,
    packed,
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


def _build_selectors(kept: list[TrainingDatum]) -> dict[str, np.ndarray]:
    phases = [phase_from_features(d.features) for d in kept]
    counts = [_piece_count(d.fen_before) for d in kept]
    kap = [_is_kings_and_pawns(d.fen_before) for d in kept]
    n = len(kept)
    return {
        "all": np.ones(n, dtype=bool),
        "opening": np.asarray([p == "opening" for p in phases], dtype=bool),
        "middle": np.asarray([p == "middle" for p in phases], dtype=bool),
        "endgame": np.asarray([p == "endgame" for p in phases], dtype=bool),
        "pieces_le_10": np.asarray([c <= 10 for c in counts], dtype=bool),
        "pieces_le_6": np.asarray([c <= 6 for c in counts], dtype=bool),
        "kings_pawns": np.asarray(kap, dtype=bool),
    }


def _shortlist_misses(
    *,
    logits: np.ndarray,
    packed,
    selector: np.ndarray,
    limit: int,
) -> list[str]:
    """SF-disagree + top1 miss within selector (Play-like failure shortlist)."""
    idx = np.flatnonzero(selector)
    if idx.size == 0:
        return []
    masked = np.where(packed.mask > 0.5, logits, -np.inf)
    top1 = np.argmax(masked, axis=1) == packed.labels
    disagree = user_not_sf_best_mask(packed.feats, packed.labels)
    lines: list[str] = []
    for i in idx:
        if top1[i] or not disagree[i]:
            continue
        d = packed.kept_datums[int(i)]
        lines.append(f"  pieces={_piece_count(d.fen_before)} ply={d.ply} fen={d.fen_before}")
        if len(lines) >= limit:
            break
    return lines


def run(
    *,
    models: list[tuple[str, str]],
    split_version: str,
    limit: int,
    full_val: bool,
    error_limit: int,
) -> int:
    db = get_db_client()
    val = load_registry_val_datums(
        db,
        split_version=split_version,
        limit=None if full_val else limit,
        full=full_val,
    )
    if len(val) < 50:
        logger.error("Val too small: %s", len(val))
        return 1
    logger.info("Packing val n=%s…", len(val))
    packed = pack_datums_for_eval(val)
    selectors = _build_selectors(packed.kept_datums)
    print("\n=== sparse / phase slice counts (kept) ===")
    for name, sel in selectors.items():
        print(f"  {name}: n={int(np.sum(sel))}")

    slice_order = list(selectors.keys())
    results: dict[str, dict[str, EvalMetrics | None]] = {}

    for label, uri in models:
        path = Path(uri)
        load_uri = str(path.resolve()) if path.is_file() else uri
        logger.info("Loading model %s uri=%s", label, load_uri)
        model = load_candidate_style_from_uri(load_uri)
        logits = predict_candidate_logits(model, packed.kept_datums, packed.feats)
        results[label] = {
            name: _metrics_on_selector(logits=logits, packed=packed, selector=sel)
            for name, sel in selectors.items()
        }
        print(f"\n=== {label} ({uri}) ===")
        for name in slice_order:
            m = results[label][name]
            if m is None:
                print(f"  {name}: (too few)")
                continue
            print(format_eval_metrics(name, m))
        print(f"\n--- {label} kings_pawns SF-disagree top1-miss shortlist ---")
        for line in _shortlist_misses(
            logits=logits,
            packed=packed,
            selector=selectors["kings_pawns"],
            limit=error_limit,
        ) or ["  (none)"]:
            print(line)
        print(f"\n--- {label} pieces_le_6 SF-disagree top1-miss shortlist ---")
        for line in _shortlist_misses(
            logits=logits,
            packed=packed,
            selector=selectors["pieces_le_6"],
            limit=error_limit,
        ) or ["  (none)"]:
            print(line)

    if len(models) >= 2:
        a_label, _ = models[0]
        b_label, _ = models[1]
        print(f"\n=== delta {b_label} - {a_label} (disagree_t1 / agree_t1) ===")
        for name in slice_order:
            ma = results[a_label][name]
            mb = results[b_label][name]
            if ma is None or mb is None:
                print(f"  {name}: (skip)")
                continue
            if ma.top1_sf_disagree is None or mb.top1_sf_disagree is None:
                d_dis = float("nan")
            else:
                d_dis = float(mb.top1_sf_disagree) - float(ma.top1_sf_disagree)
            if ma.top1_sf_agree is None or mb.top1_sf_agree is None:
                d_ag = float("nan")
            else:
                d_ag = float(mb.top1_sf_agree) - float(ma.top1_sf_agree)
            print(f"  {name}: d_disagree={d_dis:+.3f} d_agree={d_ag:+.3f} (n={mb.n_eval:.0f})")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        action="append",
        required=True,
        type=_parse_model_arg,
        help="LABEL=path_or_uri (repeatable)",
    )
    p.add_argument("--split-version", type=str, default=DEFAULT_SPLIT_SALT)
    p.add_argument("--limit", type=int, default=10000)
    p.add_argument("--full-val", action="store_true")
    p.add_argument("--error-limit", type=int, default=15)
    args = p.parse_args()
    log_script_runtime_context(logger, script="analyze_val_sparse_material")
    return run(
        models=list(args.model),
        split_version=str(args.split_version),
        limit=max(50, int(args.limit)),
        full_val=bool(args.full_val),
        error_limit=max(1, int(args.error_limit)),
    )


if __name__ == "__main__":
    run_script_main(main)
