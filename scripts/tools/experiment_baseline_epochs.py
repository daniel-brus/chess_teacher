"""Sweep Keras epochs on a fixed packed batch; report train vs held-out metrics.

Packs candidate feats once, then cold-starts a fresh model per epoch count with
an 80/20 split (validation_data). Use to pick ``BaselineTrainer.DEFAULT_EPOCHS``.

Run (dev)::

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/tools/experiment_baseline_epochs.py

Optional::

    --limit 2000
    --epochs 3,5,8,12
    --val-fraction 0.2
    --style-disagree-boost 2.0
    --style-disagree-scale 2.0
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from chess_teacher.pipelines.neural_network.create_training_set import (
    TrainingBatch,
    TrainingDataStore,
)
from chess_teacher.pipelines.neural_network.ply_weights import (
    candidate_style_sample_weights,
)
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


def _split(
    n: int,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, round(n * val_fraction))
    n_val = min(n_val, n - 1) if n > 1 else 1
    val_idx = np.sort(idx[:n_val])
    train_idx = np.sort(idx[n_val:])
    return train_idx, val_idx


def run_sweep(
    *,
    limit: int,
    epoch_grid: list[int],
    val_fraction: float,
    style_disagree_boost: float,
    style_disagree_scale: float,
    seed: int,
) -> int:
    db = get_db_client()
    # Ignore training cutoff — sweep needs a sizable fixed sample of eligible moves.
    logger.info("Loading datums limit=%s (cutoff=None for experiment sample)…", limit)
    datums, _cutoff = TrainingDataStore(db).fetch_since(None, limit=limit)
    if len(datums) < 50:
        logger.error("Need more datums for sweep; got %s", len(datums))
        return 1

    batch = TrainingBatch(datums)
    feats, mask, labels, kept = batch.candidate_style_targets()
    if len(kept) < 50:
        logger.error("Too few kept after packing: %s", len(kept))
        return 1
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

    train_idx, val_idx = _split(len(kept_datums), val_fraction=val_fraction, seed=seed)
    logger.info(
        "Packed n=%s train=%s val=%s style_disagree_boost=%s scale_pawns=%s epoch_grid=%s",
        len(kept_datums),
        len(train_idx),
        len(val_idx),
        style_disagree_boost,
        style_disagree_scale,
        epoch_grid,
    )

    x_tr = {"state": x_state[train_idx], "move_feats": feats[train_idx]}
    y_tr = y[train_idx]
    w_tr = sample_w[train_idx]
    x_va = {"state": x_state[val_idx], "move_feats": feats[val_idx]}
    y_va = y[val_idx]
    w_va = sample_w[val_idx]

    rows: list[dict[str, float]] = []
    for epochs in epoch_grid:
        trainer = BaselineTrainer(
            epochs=epochs,
            style_disagree_boost=style_disagree_boost,
            style_disagree_scale=style_disagree_scale,
        )
        model = trainer.build(input_dim=int(x_state.shape[1]))
        t0 = time.perf_counter()
        history = model.fit(
            x_tr,
            y_tr,
            sample_weight=w_tr,
            validation_data=(x_va, y_va, w_va),
            epochs=epochs,
            batch_size=min(trainer.batch_size, len(train_idx)),
            verbose=0,
        )
        dt = time.perf_counter() - t0
        h = history.history
        row = {
            "epochs": float(epochs),
            "fit_s": dt,
            "train_loss": float(h["loss"][-1]),
            "val_loss": float(h["val_loss"][-1]),
            "train_top1": float(h["masked_cand_top1"][-1]),
            "val_top1": float(h["val_masked_cand_top1"][-1]),
            "train_top3": float(h["masked_cand_top3"][-1]),
            "val_top3": float(h["val_masked_cand_top3"][-1]),
            "best_val_loss_epoch": float(int(np.argmin(h["val_loss"])) + 1),
            "best_val_top1_epoch": float(int(np.argmax(h["val_masked_cand_top1"])) + 1),
        }
        rows.append(row)
        logger.info(
            "epochs=%s fit_s=%.1f train_loss=%.4f val_loss=%.4f "
            "train_top1=%.4f val_top1=%.4f best_val_loss@%s best_val_top1@%s",
            epochs,
            dt,
            row["train_loss"],
            row["val_loss"],
            row["train_top1"],
            row["val_top1"],
            int(row["best_val_loss_epoch"]),
            int(row["best_val_top1_epoch"]),
        )

    print("\n=== epoch sweep summary ===")
    print(
        f"{'ep':>4} {'tr_loss':>8} {'va_loss':>8} {'tr_t1':>7} {'va_t1':>7} "
        f"{'gap_t1':>7} {'best_va_loss@':>12} {'best_va_t1@':>11} {'s':>6}"
    )
    for row in rows:
        gap = row["train_top1"] - row["val_top1"]
        print(
            f"{int(row['epochs']):4d} {row['train_loss']:8.4f} {row['val_loss']:8.4f} "
            f"{row['train_top1']:7.4f} {row['val_top1']:7.4f} {gap:7.4f} "
            f"{int(row['best_val_loss_epoch']):12d} {int(row['best_val_top1_epoch']):11d} "
            f"{row['fit_s']:6.1f}"
        )

    # Suggest: max epochs where val_top1 still near peak and gap not exploding.
    best_by_val = max(rows, key=lambda r: r["val_top1"])
    print(
        f"\nPeak val_top1={best_by_val['val_top1']:.4f} at epochs={int(best_by_val['epochs'])} "
        f"(within-run best epoch for that fit: {int(best_by_val['best_val_top1_epoch'])})."
    )
    print(
        "Rule of thumb: prefer the smallest epoch count within ~0.01 of peak val_top1 "
        "with modest train-val top1 gap."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--epochs", type=str, default="3,5,8,12")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--style-disagree-boost", type=float, default=2.0)
    parser.add_argument("--style-disagree-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    log_script_runtime_context(logger, script="experiment_baseline_epochs")
    return run_sweep(
        limit=max(50, args.limit),
        epoch_grid=_parse_epochs(args.epochs),
        val_fraction=min(0.5, max(0.05, args.val_fraction)),
        style_disagree_boost=float(args.style_disagree_boost),
        style_disagree_scale=float(args.style_disagree_scale),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    run_script_main(main)
