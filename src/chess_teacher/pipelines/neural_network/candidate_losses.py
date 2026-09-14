"""E22 candidate-style loss variants (masked categorical over legal slots).

Layouts (float32 ``y_true``; last column always **user** class index for metrics):

- **sparse** (A): ``[mask | user_idx]`` → ``(N, MAX+1)`` — legacy alias
- **soft** (B): ``[mask | soft_probs | user_idx]`` → ``(N, 2*MAX+1)`` — research only
- **sf_mix** (C): ``[mask | sf_idx | user_idx]`` → ``(N, MAX+2)`` — **default path**

Default training objective: ``loss_kind=sf_mix`` with ``sf_mix_alpha=0`` → pure user
CE (same goal as sparse). Raise ``alpha`` only for platform / generic baselines that
should leash toward SF-best among candidates. Personal / style bots stay at ``0``.

Soft probs come from temperature-softmax of SF ``delta_vs_best`` (user POV pawns)
among masked candidates. SF index = masked argmax of ``evaluation_after_user_pov``.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from chess_teacher.pipelines.neural_network.candidate_eval import (
    CANDIDATE_MOVE_FEAT_KEYS,
    MAX_CANDIDATES,
    MOVE_FEAT_DIM,
)
from chess_teacher.pipelines.neural_network.tf_runtime import ensure_tensorflow_logging

LossKind = Literal["sparse", "soft", "sf_mix"]

DEFAULT_LOSS_KIND: LossKind = "sf_mix"
DEFAULT_SOFT_TEMPERATURE_PAWNS = 0.5
# alpha=0 -> user-only CE (personal / style). alpha>0 -> SF leash for platform baselines.
DEFAULT_SF_MIX_ALPHA = 0.0

# Must match packing in ``candidate_eval`` / ``ply_weights``.
_EVAL_FEAT_TANH_SCALE = 5.0
_ATANH_CLIP = 0.999

_DELTA_KEY = "delta_vs_best"
_EVAL_AFTER_KEY = "evaluation_after_user_pov"


def _feat_channel(move_feats: np.ndarray, key: str) -> np.ndarray:
    feats = np.asarray(move_feats, dtype=np.float64)
    if feats.ndim != 3 or feats.shape[-1] != MOVE_FEAT_DIM:
        raise ValueError(f"move_feats expected (N, MAX, {MOVE_FEAT_DIM}), got {feats.shape}")
    try:
        idx = CANDIDATE_MOVE_FEAT_KEYS.index(key)
    except ValueError as exc:
        raise ValueError(f"missing feat key {key!r}") from exc
    return feats[:, :, idx]


def _tanh_feat_to_pawns(
    feat: np.ndarray, *, tanh_scale: float = _EVAL_FEAT_TANH_SCALE
) -> np.ndarray:
    clipped = np.clip(feat, -_ATANH_CLIP, _ATANH_CLIP)
    return float(tanh_scale) * np.arctanh(clipped)


def soft_labels_from_delta_vs_best(
    move_feats: np.ndarray,
    mask: np.ndarray,
    *,
    temperature_pawns: float = DEFAULT_SOFT_TEMPERATURE_PAWNS,
    tanh_scale: float = _EVAL_FEAT_TANH_SCALE,
) -> np.ndarray:
    """``(N, MAX)`` soft targets: softmax(delta_pawns / T) on masked slots."""
    temp = float(temperature_pawns)
    if temp <= 0:
        raise ValueError(f"temperature_pawns must be > 0, got {temp}")
    m = np.asarray(mask, dtype=np.float64)
    delta = _tanh_feat_to_pawns(_feat_channel(move_feats, _DELTA_KEY), tanh_scale=tanh_scale)
    logits = np.where(m > 0.5, delta / temp, -1.0e9)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits) * (m > 0.5)
    denom = np.sum(exp, axis=1, keepdims=True)
    denom = np.maximum(denom, 1.0e-12)
    return (exp / denom).astype(np.float32)


def sf_best_indices_from_eval(
    move_feats: np.ndarray,
    mask: np.ndarray,
    *,
    tanh_scale: float = _EVAL_FEAT_TANH_SCALE,
) -> np.ndarray:
    """Masked argmax of SF eval-after (user POV pawns) → ``(N,)`` int64."""
    m = np.asarray(mask, dtype=np.float64)
    ev = _tanh_feat_to_pawns(_feat_channel(move_feats, _EVAL_AFTER_KEY), tanh_scale=tanh_scale)
    scores = np.where(m > 0.5, ev, -1.0e9)
    return np.argmax(scores, axis=1).astype(np.int64)


def pack_sparse_candidate_targets(labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``[mask | user_idx]``."""
    y_index = np.asarray(labels, dtype=np.float32).reshape(-1, 1)
    m = np.asarray(mask, dtype=np.float32)
    return np.concatenate([m, y_index], axis=1)


def pack_soft_candidate_targets(
    labels: np.ndarray,
    mask: np.ndarray,
    soft: np.ndarray,
) -> np.ndarray:
    """``[mask | soft | user_idx]``."""
    m = np.asarray(mask, dtype=np.float32)
    s = np.asarray(soft, dtype=np.float32)
    if s.shape != m.shape:
        raise ValueError(f"soft shape {s.shape} != mask shape {m.shape}")
    y_index = np.asarray(labels, dtype=np.float32).reshape(-1, 1)
    return np.concatenate([m, s, y_index], axis=1)


def pack_sf_mix_candidate_targets(
    labels: np.ndarray,
    mask: np.ndarray,
    sf_indices: np.ndarray,
) -> np.ndarray:
    """``[mask | sf_idx | user_idx]``."""
    m = np.asarray(mask, dtype=np.float32)
    sf = np.asarray(sf_indices, dtype=np.float32).reshape(-1, 1)
    y_index = np.asarray(labels, dtype=np.float32).reshape(-1, 1)
    if sf.shape[0] != m.shape[0]:
        raise ValueError(f"sf_indices length {sf.shape[0]} != N {m.shape[0]}")
    return np.concatenate([m, sf, y_index], axis=1)


def pack_candidate_targets_for_loss(
    *,
    loss_kind: LossKind,
    labels: np.ndarray,
    mask: np.ndarray,
    move_feats: np.ndarray,
    soft_temperature_pawns: float = DEFAULT_SOFT_TEMPERATURE_PAWNS,
) -> np.ndarray:
    """Pack ``y_true`` for the chosen E22 loss kind."""
    kind = str(loss_kind)
    if kind == "sparse":
        return pack_sparse_candidate_targets(labels, mask)
    if kind == "soft":
        soft = soft_labels_from_delta_vs_best(
            move_feats, mask, temperature_pawns=soft_temperature_pawns
        )
        return pack_soft_candidate_targets(labels, mask, soft)
    if kind == "sf_mix":
        sf_idx = sf_best_indices_from_eval(move_feats, mask)
        return pack_sf_mix_candidate_targets(labels, mask, sf_idx)
    raise ValueError(f"unknown loss_kind={loss_kind!r}")


def _import_tf() -> Any:
    ensure_tensorflow_logging()
    import tensorflow as tf  # type: ignore[import-untyped]

    return tf


def masked_candidate_sparse_ce(max_candidates: int = MAX_CANDIDATES):
    """Variant A -- one-hot user index (``y_true`` last col)."""
    tf = _import_tf()

    def loss_fn(y_true: Any, y_pred: Any) -> Any:
        mask = y_true[:, :max_candidates]
        indices = tf.cast(y_true[:, -1], tf.int32)
        neg_inf = tf.constant(-1.0e9, dtype=y_pred.dtype)
        masked_logits = tf.where(mask > 0.5, y_pred, neg_inf)
        return tf.keras.losses.sparse_categorical_crossentropy(
            indices, masked_logits, from_logits=True
        )

    loss_fn.__name__ = "masked_candidate_sparse_ce"
    return loss_fn


def masked_candidate_soft_ce(max_candidates: int = MAX_CANDIDATES):
    """Variant B -- soft SF distribution among masked slots."""
    tf = _import_tf()

    def loss_fn(y_true: Any, y_pred: Any) -> Any:
        mask = y_true[:, :max_candidates]
        soft = y_true[:, max_candidates : 2 * max_candidates]
        neg_inf = tf.constant(-1.0e9, dtype=y_pred.dtype)
        masked_logits = tf.where(mask > 0.5, y_pred, neg_inf)
        log_probs = tf.nn.log_softmax(masked_logits, axis=-1)
        # soft already ~0 on illegal slots
        return -tf.reduce_sum(soft * log_probs, axis=-1)

    loss_fn.__name__ = "masked_candidate_soft_ce"
    return loss_fn


def masked_candidate_sf_mix_ce(
    max_candidates: int = MAX_CANDIDATES,
    *,
    alpha: float = DEFAULT_SF_MIX_ALPHA,
):
    """Default path -- ``(1-alpha)*CE_user + alpha*CE_sf_best`` (alpha=0 == user-only)."""
    tf = _import_tf()
    a = float(alpha)
    if not 0.0 <= a <= 1.0:
        raise ValueError(f"sf_mix alpha must be in [0,1], got {a}")

    def loss_fn(y_true: Any, y_pred: Any) -> Any:
        mask = y_true[:, :max_candidates]
        sf_idx = tf.cast(y_true[:, max_candidates], tf.int32)
        user_idx = tf.cast(y_true[:, -1], tf.int32)
        neg_inf = tf.constant(-1.0e9, dtype=y_pred.dtype)
        masked_logits = tf.where(mask > 0.5, y_pred, neg_inf)
        ce_user = tf.keras.losses.sparse_categorical_crossentropy(
            user_idx, masked_logits, from_logits=True
        )
        ce_sf = tf.keras.losses.sparse_categorical_crossentropy(
            sf_idx, masked_logits, from_logits=True
        )
        return (1.0 - a) * ce_user + a * ce_sf

    loss_fn.__name__ = "masked_candidate_sf_mix_ce"
    return loss_fn


def masked_candidate_top_k(k: int, max_candidates: int = MAX_CANDIDATES):
    """Top-k vs **user** index (always last column of ``y_true``)."""
    tf = _import_tf()

    def metric_fn(y_true: Any, y_pred: Any) -> Any:
        mask = y_true[:, :max_candidates]
        indices = tf.cast(y_true[:, -1], tf.int32)
        neg_inf = tf.constant(-1.0e9, dtype=y_pred.dtype)
        masked_logits = tf.where(mask > 0.5, y_pred, neg_inf)
        return tf.keras.metrics.sparse_top_k_categorical_accuracy(indices, masked_logits, k=k)

    metric_fn.__name__ = f"masked_cand_top{k}"
    return metric_fn


def resolve_candidate_loss(
    loss_kind: LossKind = DEFAULT_LOSS_KIND,
    *,
    max_candidates: int = MAX_CANDIDATES,
    sf_mix_alpha: float = DEFAULT_SF_MIX_ALPHA,
) -> Any:
    kind = str(loss_kind)
    if kind == "sparse":
        return masked_candidate_sparse_ce(max_candidates)
    if kind == "soft":
        return masked_candidate_soft_ce(max_candidates)
    if kind == "sf_mix":
        return masked_candidate_sf_mix_ce(max_candidates, alpha=sf_mix_alpha)
    raise ValueError(f"unknown loss_kind={loss_kind!r}")


def candidate_loss_custom_objects(
    max_candidates: int = MAX_CANDIDATES,
    *,
    sf_mix_alpha: float = DEFAULT_SF_MIX_ALPHA,
) -> dict[str, Any]:
    """Keras ``custom_objects`` for save/load (all E22 names registered)."""
    return {
        "masked_candidate_sparse_ce": masked_candidate_sparse_ce(max_candidates),
        "masked_candidate_soft_ce": masked_candidate_soft_ce(max_candidates),
        "masked_candidate_sf_mix_ce": masked_candidate_sf_mix_ce(
            max_candidates, alpha=sf_mix_alpha
        ),
        "masked_cand_top1": masked_candidate_top_k(1, max_candidates),
        "masked_cand_top3": masked_candidate_top_k(3, max_candidates),
    }
