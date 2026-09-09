"""Sample weights: ply reweight + SF-style, recency, and baseline-disagree knobs.

Ply: ``w ∝ exp(λ · ply)`` (opening-heavy data).

Style / recency / baseline-disagree (same multiplier shape):
    w *= 1 + (BOOST - 1) * strength

- SF-style: ``strength = clip(-delta_vs_best_pawns / SCALE, 0, 1)``
- Recency: ``strength = exp(-λ · age_days)``; missing ``end_time`` -> 0
- Baseline-disagree v1: hard mask 1 if user label ≠ frozen production
  baseline masked-argmax (not the last personal-bot checkpoint)

``delta_vs_best`` is recovered from the packed tanh feat
    (``tanh(delta / EVAL_TANH_SCALE)``).

Knobs (env, optional; baseline SF-style only):
- ``BASELINE_STYLE_DISAGREE_BOOST`` — max multiplier (default ``2.0``); ``1.0`` = off
- ``BASELINE_STYLE_DISAGREE_SCALE`` — pawns of Δ to reach max (default ``2.0``)

``BaselineTrainer.baseline_disagree_boost`` defaults to ``1.0`` (off). User CLIs
pass ``USER_BASELINE_DISAGREE_BOOST`` (4.0). Do not import that constant into
platform baseline catch-up.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np

from chess_teacher.pipelines.neural_network.candidate_eval import CANDIDATE_MOVE_FEAT_KEYS
from chess_teacher.utils.env_utils import get_optional_env_variable
from chess_teacher.utils.general_utils import as_utc, get_current_datetime
from chess_teacher.utils.logging import get_logger

logger = get_logger()

DEFAULT_PLY_WEIGHT_LAMBDA = 0.016
DEFAULT_PLY_WEIGHT_CLIP = (0.25, 4.0)
# Max weight multiplier at full disagreement (vs SF-best). 1.0 = style term off.
DEFAULT_STYLE_DISAGREE_BOOST = 2.0
# |delta_vs_best| in pawns that maps to strength=1 (hard cap on the ramp).
DEFAULT_STYLE_DISAGREE_SCALE = 2.0
# Recency: w *= 1 + (BOOST-1)*exp(-lam*age_days). 1.0 = recency off.
DEFAULT_RECENCY_BOOST = 2.0
DEFAULT_RECENCY_LAMBDA = 0.02
# User-CLI baseline-disagree hard mask. Trainer default stays 1.0 (off).
USER_BASELINE_DISAGREE_BOOST = 4.0
# Must match candidate_eval packing for evaluation / delta_vs_best.
_EVAL_FEAT_TANH_SCALE = 5.0
_DELTA_VS_BEST_KEY = "delta_vs_best"
_DELTA_VS_BEST_EPS = 1e-5
_ATANH_CLIP = 0.999


def _env_float(name: str, default: float, *, min_value: float | None = None) -> float:
    raw = get_optional_env_variable(name)
    if not raw:
        return float(default)
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return float(default)
    if min_value is not None and value < min_value:
        logger.warning(
            "%s=%s below min %s; using default %s",
            name,
            value,
            min_value,
            default,
        )
        return float(default)
    return value


def style_disagree_boost_from_env(
    default: float = DEFAULT_STYLE_DISAGREE_BOOST,
) -> float:
    """Read ``BASELINE_STYLE_DISAGREE_BOOST`` (≥0); invalid -> default."""
    return _env_float("BASELINE_STYLE_DISAGREE_BOOST", default, min_value=0.0)


def style_disagree_scale_from_env(
    default: float = DEFAULT_STYLE_DISAGREE_SCALE,
) -> float:
    """Read ``BASELINE_STYLE_DISAGREE_SCALE`` (>0 pawns); invalid -> default."""
    return _env_float("BASELINE_STYLE_DISAGREE_SCALE", default, min_value=1e-6)


def ply_weight_raw(
    ply: int | list[int] | np.ndarray,
    *,
    lam: float = DEFAULT_PLY_WEIGHT_LAMBDA,
) -> np.ndarray:
    """Unnormalized ``exp(λ · ply)``."""
    p = np.asarray(ply, dtype=np.float64)
    return np.exp(lam * p)


def normalize_sample_weights(
    weights: np.ndarray,
    *,
    clip: tuple[float, float] | None = DEFAULT_PLY_WEIGHT_CLIP,
) -> np.ndarray:
    """Mean-normalize to 1; optional clip then re-normalize."""
    w = np.asarray(weights, dtype=np.float64)
    if w.size == 0:
        return w.astype(np.float32)
    mean = float(np.mean(w))
    if mean <= 0:
        return np.ones_like(w, dtype=np.float32)
    w = w / mean
    if clip is not None:
        lo, hi = clip
        w = np.clip(w, lo, hi)
        mean2 = float(np.mean(w))
        if mean2 > 0:
            w = w / mean2
    return w.astype(np.float32)


def ply_sample_weights(
    plies: list[int] | np.ndarray,
    *,
    lam: float = DEFAULT_PLY_WEIGHT_LAMBDA,
    clip: tuple[float, float] | None = DEFAULT_PLY_WEIGHT_CLIP,
) -> np.ndarray:
    """Batch sample weights for train/eval (mean ≈ 1 after normalize/clip)."""
    return normalize_sample_weights(ply_weight_raw(plies, lam=lam), clip=clip)


def _labeled_delta_feat(
    move_feats: np.ndarray,
    labels: np.ndarray,
    *,
    delta_key: str = _DELTA_VS_BEST_KEY,
) -> np.ndarray:
    feats = np.asarray(move_feats, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if feats.ndim != 3:
        raise ValueError(f"move_feats expected (N, MAX, F), got {feats.shape}")
    if y.shape[0] != feats.shape[0]:
        raise ValueError(f"labels length {y.shape[0]} != N {feats.shape[0]}")
    try:
        delta_i = CANDIDATE_MOVE_FEAT_KEYS.index(delta_key)
    except ValueError as exc:
        raise ValueError(f"missing feat key {delta_key!r} in CANDIDATE_MOVE_FEAT_KEYS") from exc
    rows = np.arange(feats.shape[0], dtype=np.int64)
    return feats[rows, y, delta_i]


def labeled_delta_vs_best_pawns(
    move_feats: np.ndarray,
    labels: np.ndarray,
    *,
    tanh_scale: float = _EVAL_FEAT_TANH_SCALE,
) -> np.ndarray:
    """Invert packed ``tanh(delta/scale)`` back to approx pawns (user POV)."""
    feat = _labeled_delta_feat(move_feats, labels)
    clipped = np.clip(feat, -_ATANH_CLIP, _ATANH_CLIP)
    return float(tanh_scale) * np.arctanh(clipped)


def user_sf_disagree_strength(
    move_feats: np.ndarray,
    labels: np.ndarray,
    *,
    scale_pawns: float,
) -> np.ndarray:
    """Continuous strength in ``[0, 1]`` from how far user is below SF-best.

    ``strength = clip(-delta_pawns / scale_pawns, 0, 1)``.
    """
    scale = float(scale_pawns)
    if scale <= 0:
        raise ValueError(f"scale_pawns must be > 0, got {scale}")
    delta = labeled_delta_vs_best_pawns(move_feats, labels)
    return np.clip(-delta / scale, 0.0, 1.0)


def user_not_sf_best_mask(
    move_feats: np.ndarray,
    labels: np.ndarray,
    *,
    eps: float = _DELTA_VS_BEST_EPS,
) -> np.ndarray:
    """True where labeled move is clearly worse than SF-best (packed delta < 0)."""
    return _labeled_delta_feat(move_feats, labels) < -float(eps)


def candidate_style_sample_weights(
    plies: list[int] | np.ndarray,
    move_feats: np.ndarray,
    labels: np.ndarray,
    *,
    style_disagree_boost: float | None = None,
    style_disagree_scale: float | None = None,
    lam: float = DEFAULT_PLY_WEIGHT_LAMBDA,
    clip: tuple[float, float] | None = DEFAULT_PLY_WEIGHT_CLIP,
) -> np.ndarray:
    """Ply weights * continuous style boost; mean-normalized (clip).

    ``style_disagree_boost`` / ``style_disagree_scale`` ``None`` -> env / defaults.
    Boost ``1.0`` disables the style term (ply only).
    """
    boost = (
        style_disagree_boost_from_env()
        if style_disagree_boost is None
        else float(style_disagree_boost)
    )
    scale = (
        style_disagree_scale_from_env()
        if style_disagree_scale is None
        else float(style_disagree_scale)
    )
    raw = ply_weight_raw(plies, lam=lam)
    if boost != 1.0:
        strength = user_sf_disagree_strength(move_feats, labels, scale_pawns=scale)
        raw = raw * (1.0 + (boost - 1.0) * strength)
    return normalize_sample_weights(raw, clip=clip)


def recency_weight_raw(
    end_times: Sequence[datetime | None],
    *,
    lam: float,
    now: datetime,
) -> np.ndarray:
    """Recency strength ``exp(-lam * age_days)`` in ``[0, 1]``.

    Naive datetimes treated as UTC. Missing ``end_time`` (None) -> 0.0
    (not newest). ``lam=0`` -> 1.0 for dated rows.
    """
    items = list(end_times)
    n = len(items)
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    now_utc = as_utc(now)
    recency = np.empty(n, dtype=np.float64)
    lam_f = float(lam)
    for i, end in enumerate(items):
        if end is None:
            recency[i] = 0.0
            continue
        if lam_f == 0.0:
            recency[i] = 1.0
            continue
        age_days = max(0.0, (now_utc - as_utc(end)).total_seconds() / 86400.0)
        recency[i] = float(np.exp(-lam_f * age_days))
    return recency


def baseline_disagree_strength(
    logits: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Hard mask v1: 1 if user label != baseline masked-argmax else 0."""
    logits_arr = np.asarray(logits, dtype=np.float64)
    mask_arr = np.asarray(mask, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if logits_arr.ndim != 2:
        raise ValueError(f"logits expected (N, MAX), got {logits_arr.shape}")
    if mask_arr.shape != logits_arr.shape:
        raise ValueError(f"mask shape {mask_arr.shape} != logits shape {logits_arr.shape}")
    if y.shape[0] != logits_arr.shape[0]:
        raise ValueError(f"labels length {y.shape[0]} != N {logits_arr.shape[0]}")
    masked = np.where(mask_arr > 0.5, logits_arr, -1.0e9)
    pred = np.argmax(masked, axis=1)
    return (pred != y).astype(np.float64)


def user_finetune_sample_weights(
    plies: list[int] | np.ndarray,
    move_feats: np.ndarray,
    labels: np.ndarray,
    end_times: Sequence[datetime | None],
    *,
    recency_lambda: float,
    recency_boost: float = DEFAULT_RECENCY_BOOST,
    style_disagree_boost: float | None = None,
    style_disagree_scale: float | None = None,
    baseline_disagree_strength: np.ndarray | None = None,
    baseline_disagree_boost: float = 1.0,
    lam: float = DEFAULT_PLY_WEIGHT_LAMBDA,
    clip: tuple[float, float] | None = DEFAULT_PLY_WEIGHT_CLIP,
    now: datetime | None = None,
) -> np.ndarray:
    """Ply x SF (clip), then capped recency + baseline-disagree; mean-norm no clip.

    Recency: ``w *= 1 + (BOOST-1)*exp(-lam*age_days)``. Boost ``1.0`` disables.
    Baseline-disagree: ``w *= 1 + (BOOST-1)*strength`` with hard mask v1 vs the
    frozen production baseline for this lineage. Boost ``1.0`` disables.
    ``recency_boost=1`` and ``baseline_disagree_boost=1`` matches
    ``candidate_style_sample_weights``.
    """
    boost = (
        style_disagree_boost_from_env()
        if style_disagree_boost is None
        else float(style_disagree_boost)
    )
    scale = (
        style_disagree_scale_from_env()
        if style_disagree_scale is None
        else float(style_disagree_scale)
    )
    raw = ply_weight_raw(plies, lam=lam)
    if boost != 1.0:
        strength = user_sf_disagree_strength(move_feats, labels, scale_pawns=scale)
        raw = raw * (1.0 + (boost - 1.0) * strength)
    # Clip ply x SF first so recency/baseline-disagree tails are not floored to 0.25.
    styled = normalize_sample_weights(raw, clip=clip)
    w = np.asarray(styled, dtype=np.float64)
    recency_b = float(recency_boost)
    if recency_b != 1.0:
        recency_now = now if now is not None else get_current_datetime()
        recency = recency_weight_raw(end_times, lam=float(recency_lambda), now=recency_now)
        if recency.shape[0] != w.shape[0]:
            raise ValueError(
                f"end_times length {recency.shape[0]} != plies/raw length {w.shape[0]}"
            )
        w = w * (1.0 + (recency_b - 1.0) * recency)
    baseline_b = float(baseline_disagree_boost)
    if baseline_b != 1.0:
        if baseline_disagree_strength is None:
            raise ValueError("baseline_disagree_boost != 1.0 requires baseline_disagree_strength")
        baseline_s = np.asarray(baseline_disagree_strength, dtype=np.float64).reshape(-1)
        if baseline_s.shape[0] != w.shape[0]:
            raise ValueError(
                f"baseline_disagree_strength length {baseline_s.shape[0]} != N {w.shape[0]}"
            )
        w = w * (1.0 + (baseline_b - 1.0) * baseline_s)
    return normalize_sample_weights(w, clip=None)
