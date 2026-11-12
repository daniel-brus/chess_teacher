"""Chunked hybrid val scoring helpers (log-loss + metrics)."""

from __future__ import annotations

from typing import Any

import numpy as np

from chess_teacher.pipelines.neural_network.create_training_set import TrainingDatum
from chess_teacher.pipelines.neural_network.eval_metrics import (
    EVAL_PACK_CHUNK_SIZE,
    EvalMetrics,
    evaluate_datums,
    pack_datums_for_eval,
    predict_candidate_logits,
)
from chess_teacher.utils.logging import get_logger

logger = get_logger()


def masked_candidate_log_loss(
    logits: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Mean sparse CE over masked candidate slots (numpy)."""
    logits_arr = np.asarray(logits, dtype=np.float64)
    mask_arr = np.asarray(mask, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    if logits_arr.ndim != 2:
        raise ValueError(f"logits must be 2D, got {logits_arr.shape}")
    masked = np.where(mask_arr > 0.5, logits_arr, -1.0e9)
    row_max = np.max(masked, axis=1, keepdims=True)
    exp = np.exp(masked - row_max)
    # Zero illegal mass already via -1e9; still safe to multiply by mask.
    exp = exp * (mask_arr > 0.5)
    denom = np.sum(exp, axis=1, keepdims=True)
    denom = np.maximum(denom, 1.0e-12)
    log_probs = (masked - row_max) - np.log(denom)
    rows = np.arange(labels_arr.shape[0])
    return float(-np.mean(log_probs[rows, labels_arr]))


def score_hybrid_val_log_loss(
    model: Any,
    datums: list[TrainingDatum],
    *,
    chunk_size: int = EVAL_PACK_CHUNK_SIZE,
) -> float:
    """Mean masked candidate log-loss on ``datums`` (chunked pack)."""
    if not datums:
        raise ValueError("score_hybrid_val_log_loss requires non-empty datums")
    chunk_size = max(1, int(chunk_size))
    loss_sum = 0.0
    n_kept = 0
    for start in range(0, len(datums), chunk_size):
        chunk = datums[start : start + chunk_size]
        try:
            packed = pack_datums_for_eval(chunk)
        except ValueError:
            continue
        logits = predict_candidate_logits(model, packed.kept_datums, packed.feats)
        loss_sum += masked_candidate_log_loss(logits, packed.mask, packed.labels) * float(
            len(packed.labels)
        )
        n_kept += len(packed.labels)
        del packed, logits
    if n_kept < 1:
        raise ValueError("score_hybrid_val_log_loss: no usable candidate rows")
    return float(loss_sum / float(n_kept))


def score_hybrid_datums(
    model: Any,
    datums: list[TrainingDatum],
    *,
    chunk_size: int = EVAL_PACK_CHUNK_SIZE,
) -> tuple[EvalMetrics, float]:
    """Stratified metrics + mean log-loss (chunked)."""
    metrics = evaluate_datums(model, datums, chunk_size=chunk_size)
    log_loss = score_hybrid_val_log_loss(model, datums, chunk_size=chunk_size)
    return metrics, log_loss


def blend_candidate_logits(
    parent_logits: Any,
    child_logits: Any,
    *,
    alpha: float,
) -> np.ndarray:
    """``(1-alpha)*parent + alpha*child`` on candidate logits (alpha = child weight)."""
    a = float(alpha)
    if a < 0.0 or a > 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    parent = np.asarray(parent_logits, dtype=np.float64)
    child = np.asarray(child_logits, dtype=np.float64)
    if parent.shape != child.shape:
        raise ValueError(f"logit shape mismatch parent={parent.shape} child={child.shape}")
    return (1.0 - a) * parent + a * child


def score_hybrid_logit_blend(
    parent_model: Any,
    child_model: Any,
    datums: list[TrainingDatum],
    *,
    alpha: float,
    chunk_size: int = EVAL_PACK_CHUNK_SIZE,
) -> tuple[EvalMetrics, float]:
    """Stratified metrics + log-loss for a parent/child logit blend.

    Packs each chunk once, predicts both models, blends logits, then scores.
    ``alpha=0`` is parent-only; ``alpha=1`` is child-only.
    """
    from chess_teacher.pipelines.neural_network.eval_metrics import (
        _metrics_from_hit_buffers,
        _predict_packed_logits,
        _topk_hits,
        candidate_style_weight_factors,
        normalize_sample_weights,
        pack_datums_for_eval,
        predicted_sf_delta_pawns,
        user_not_sf_best_mask,
    )

    if not datums:
        raise ValueError("score_hybrid_logit_blend requires non-empty datums")
    a = float(alpha)
    if a < 0.0 or a > 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    chunk_size = max(1, int(chunk_size))
    n_input = len(datums)
    top1_parts: list[np.ndarray] = []
    top3_parts: list[np.ndarray] = []
    delta_parts: list[np.ndarray] = []
    disagree_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    loss_sum = 0.0
    n_kept = 0

    n_chunks = (n_input + chunk_size - 1) // chunk_size
    for chunk_i, start in enumerate(range(0, n_input, chunk_size)):
        chunk = datums[start : start + chunk_size]
        try:
            packed = pack_datums_for_eval(chunk)
        except ValueError:
            logger.info(
                "Blend eval chunk %s/%s dropped entirely (n_in=%s)",
                chunk_i + 1,
                n_chunks,
                len(chunk),
            )
            continue
        parent_logits = _predict_packed_logits(parent_model, packed)
        if a <= 0.0:
            logits = np.asarray(parent_logits, dtype=np.float64)
        elif a >= 1.0:
            logits = np.asarray(_predict_packed_logits(child_model, packed), dtype=np.float64)
        else:
            child_logits = _predict_packed_logits(child_model, packed)
            logits = blend_candidate_logits(parent_logits, child_logits, alpha=a)
        top1, top3 = _topk_hits(
            logits,
            packed.mask,
            packed.labels,
            max_candidates=packed.max_candidates,
        )
        top1_parts.append(np.asarray(top1, dtype=bool))
        top3_parts.append(np.asarray(top3, dtype=bool))
        delta_parts.append(predicted_sf_delta_pawns(logits, packed.mask, packed.feats))
        disagree_parts.append(
            np.asarray(user_not_sf_best_mask(packed.feats, packed.labels), dtype=bool)
        )
        weight_parts.append(
            np.asarray(
                candidate_style_weight_factors(
                    [d.ply for d in packed.kept_datums],
                    packed.feats,
                    packed.labels,
                ),
                dtype=np.float64,
            )
        )
        n_row = len(packed.labels)
        loss_sum += masked_candidate_log_loss(logits, packed.mask, packed.labels) * float(n_row)
        n_kept += n_row
        logger.info(
            "Blend eval chunk %s/%s kept=%s / chunk_in=%s alpha=%.2f",
            chunk_i + 1,
            n_chunks,
            n_row,
            len(chunk),
            a,
        )
        del packed, parent_logits, logits

    if n_kept < 1 or not disagree_parts:
        raise ValueError("score_hybrid_logit_blend: no usable candidate rows")
    disagree_all = np.concatenate(disagree_parts)
    weights_all = np.asarray(
        normalize_sample_weights(np.concatenate(weight_parts)),
        dtype=np.float64,
    )
    metrics = _metrics_from_hit_buffers(
        top1=np.concatenate(top1_parts),
        top3=np.concatenate(top3_parts),
        disagree=disagree_all,
        weights=weights_all,
        sf_delta_pawns=np.concatenate(delta_parts),
        n_input=n_input,
    )
    return metrics, float(loss_sum / float(n_kept))
