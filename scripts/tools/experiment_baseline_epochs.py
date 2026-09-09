"""Sweep Keras epochs on a frozen registry split; report stratified val metrics.

Cold-starts a fresh model per epoch count. Train games stay in train; val is the
persistent registry val subset of the loaded sample (not move-level 80/20).

Use to pick ``BaselineTrainer.DEFAULT_EPOCHS``. Smoke: ``--limit 2000``.
Decisions: ``--limit 10000+``. Does **not** touch production pipelines.

Run (dev)::

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/tools/experiment_baseline_epochs.py

Optional::

    --limit 10000
    --epochs 3,5,8,12
    --salt baseline-v1
    --style-disagree-boost 2.0
    --style-disagree-scale 2.0
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from chess_teacher.pipelines.neural_network.candidate_eval import MAX_CANDIDATES
from chess_teacher.pipelines.neural_network.create_training_set import (
    TrainingBatch,
    TrainingDatum,
)
from chess_teacher.pipelines.neural_network.eval_metrics import (
    compute_candidate_style_metrics,
    format_eval_metrics,
)
from chess_teacher.pipelines.neural_network.offline_eval import load_registry_split
from chess_teacher.pipelines.neural_network.ply_weights import candidate_style_sample_weights
from chess_teacher.pipelines.neural_network.splits import DEFAULT_SPLIT_SALT, format_split_summary
from chess_teacher.pipelines.neural_network.tf_runtime import ensure_tensorflow_logging
from chess_teacher.pipelines.neural_network.train import (
    BaselineTrainer,
    pack_candidate_targets,
)
from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.process_utils import log_script_runtime_context, run_script_main

ensure_tensorflow_logging()
logger = get_logger()


def _parse_epochs(raw: str) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        n = int(part)
        if n < 1:
            raise ValueError(f"epochs must be >= 1, got {n}")
        out.append(n)
    if not out:
        raise ValueError("empty --epochs")
    return out


def _pack_datums(
    datums: list[TrainingDatum],
    *,
    style_disagree_boost: float,
    style_disagree_scale: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[TrainingDatum]]:
    batch = TrainingBatch(datums)
    feats, mask, labels, kept = batch.candidate_style_targets()
    if not kept:
        raise ValueError("no datums with usable candidate_evaluations")
    kept_datums = [datums[i] for i in kept]
    x_state = TrainingBatch(kept_datums).state_matrix()
    y = pack_candidate_targets(labels, mask)
    sample_w = candidate_style_sample_weights(
        [d.ply for d in kept_datums],
        feats,
        labels,
        style_disagree_boost=style_disagree_boost,
        style_disagree_scale=style_disagree_scale,
    )
    x = {"state": x_state, "move_feats": feats}
    return x, y, sample_w, kept_datums


def run_sweep(
    *,
    limit: int,
    epoch_grid: list[int],
    salt: str,
    style_disagree_boost: float,
    style_disagree_scale: float,
) -> int:
    db = get_db_client()
    logger.info("Loading datums limit=%s (cutoff=None for experiment sample)…", limit)
    split = load_registry_split(db, limit=limit, split_version=salt)
    print("\n" + format_split_summary(split))

    train = split.train_datums
    val = split.val_datums
    if len(train) < 30:
        logger.error("Train split too small: %s moves", len(train))
        return 1
    if len(val) < 10:
        logger.error("Val split too small: %s moves", len(val))
        return 1

    try:
        x_tr, y_tr, w_tr, kept_train = _pack_datums(
            train,
            style_disagree_boost=style_disagree_boost,
            style_disagree_scale=style_disagree_scale,
        )
        x_va, y_va, w_va, kept_val = _pack_datums(
            val,
            style_disagree_boost=style_disagree_boost,
            style_disagree_scale=style_disagree_scale,
        )
    except ValueError as exc:
        logger.error("Packing failed: %s", exc)
        return 1

    logger.info(
        "Packed train=%s val=%s style_disagree_boost=%s scale_pawns=%s epoch_grid=%s",
        len(kept_train),
        len(kept_val),
        style_disagree_boost,
        style_disagree_scale,
        epoch_grid,
    )

    va_mask = y_va[:, :MAX_CANDIDATES]
    va_labels = np.asarray(y_va[:, MAX_CANDIDATES], dtype=np.int64)
    va_plies = [d.ply for d in kept_val]

    rows: list[dict[str, float]] = []
    for epochs in epoch_grid:
        trainer = BaselineTrainer(
            epochs=epochs,
            style_disagree_boost=style_disagree_boost,
            style_disagree_scale=style_disagree_scale,
        )
        model = trainer.build(input_dim=int(x_tr["state"].shape[1]))
        t0 = time.perf_counter()
        history = model.fit(
            x_tr,
            y_tr,
            sample_weight=w_tr,
            validation_data=(x_va, y_va, w_va),
            epochs=epochs,
            batch_size=min(trainer.batch_size, len(kept_train)),
            verbose=0,
        )
        dt = time.perf_counter() - t0
        h = history.history
        logits = np.asarray(model.predict(x_va, verbose=0), dtype=np.float64)
        val_metrics = compute_candidate_style_metrics(
            logits=logits,
            mask=va_mask,
            labels=va_labels,
            move_feats=x_va["move_feats"],
            plies=va_plies,
            n_input=len(val),
            max_candidates=MAX_CANDIDATES,
        )
        disagree_t1 = (
            float(val_metrics.top1_sf_disagree)
            if val_metrics.top1_sf_disagree is not None
            else float("nan")
        )
        row = {
            "epochs": float(epochs),
            "fit_s": dt,
            "train_loss": float(h["loss"][-1]),
            "val_loss": float(h["val_loss"][-1]),
            "train_top1": float(h["masked_cand_top1"][-1]),
            "val_top1": float(val_metrics.top1_overall),
            "val_disagree_t1": disagree_t1,
            "best_val_loss_epoch": float(int(np.argmin(h["val_loss"])) + 1),
            "best_val_top1_epoch": float(int(np.argmax(h["val_masked_cand_top1"])) + 1),
        }
        rows.append(row)
        logger.info(
            "epochs=%s fit_s=%.1f train_loss=%.4f val_loss=%.4f "
            "train_top1=%.4f val_top1=%.4f val_disagree_t1=%.4f",
            epochs,
            dt,
            row["train_loss"],
            row["val_loss"],
            row["train_top1"],
            row["val_top1"],
            row["val_disagree_t1"],
        )
        print(format_eval_metrics(f"val@{epochs}", val_metrics))

    print("\n=== epoch sweep summary (registry val) ===")
    print(
        f"{'ep':>4} {'tr_loss':>8} {'va_loss':>8} {'tr_t1':>7} {'va_t1':>7} "
        f"{'va_dis':>7} {'gap_t1':>7} {'best_va_loss@':>12} {'s':>6}"
    )
    for row in rows:
        gap = row["train_top1"] - row["val_top1"]
        print(
            f"{int(row['epochs']):4d} {row['train_loss']:8.4f} {row['val_loss']:8.4f} "
            f"{row['train_top1']:7.4f} {row['val_top1']:7.4f} "
            f"{row['val_disagree_t1']:7.4f} {gap:7.4f} "
            f"{int(row['best_val_loss_epoch']):12d} {row['fit_s']:6.1f}"
        )

    finite = [r for r in rows if np.isfinite(r["val_disagree_t1"])]
    best_by_dis = max(finite, key=lambda r: r["val_disagree_t1"]) if finite else None
    best_by_val = max(rows, key=lambda r: r["val_top1"])
    print(f"\nPeak val_top1={best_by_val['val_top1']:.4f} at epochs={int(best_by_val['epochs'])}.")
    if best_by_dis is not None:
        print(
            f"Peak val disagree_t1={best_by_dis['val_disagree_t1']:.4f} "
            f"at epochs={int(best_by_dis['epochs'])}."
        )
    print(
        "Reference (not locked): 10k cold 128/64, 3 epochs, val disagree_t1~=0.20. "
        "Prefer smallest epoch count near peak val disagree_t1 with modest train-val top1 gap."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--epochs", type=str, default="3,5,8,12")
    parser.add_argument("--salt", type=str, default=DEFAULT_SPLIT_SALT)
    parser.add_argument("--style-disagree-boost", type=float, default=2.0)
    parser.add_argument("--style-disagree-scale", type=float, default=2.0)
    args = parser.parse_args()
    log_script_runtime_context(logger, script="experiment_baseline_epochs")
    return run_sweep(
        limit=max(50, args.limit),
        epoch_grid=_parse_epochs(args.epochs),
        salt=str(args.salt),
        style_disagree_boost=float(args.style_disagree_boost),
        style_disagree_scale=float(args.style_disagree_scale),
    )


if __name__ == "__main__":
    run_script_main(main)
