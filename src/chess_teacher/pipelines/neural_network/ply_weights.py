"""Sample weights: ply reweight + continuous SF-disagree style knob.

Ply: ``w ∝ exp(λ · ply)`` (opening-heavy data).

Style (continuous, capped):
    strength = clip( -delta_vs_best_pawns / SCALE , 0, 1 )
    w *= 1 + (BOOST_MAX - 1) * strength

``delta_vs_best`` is recovered from the packed tanh feat
(``tanh(delta / EVAL_TANH_SCALE)``).

Knobs (env, optional):
- ``BASELINE_STYLE_DISAGREE_BOOST`` — max multiplier (default ``2.0``); ``1.0`` = off
- ``BASELINE_STYLE_DISAGREE_SCALE`` — pawns of Δ to reach max (default ``2.0``)
"""

from __future__ import annotations

import numpy as np

from chess_teacher.pipelines.neural_network.candidate_eval import CANDIDATE_MOVE_FEAT_KEYS
from chess_teacher.utils.env_utils import get_optional_env_variable
from chess_teacher.utils.logging import get_logger

logger = get_logger()

DEFAULT_PLY_WEIGHT_LAMBDA = 0.016
DEFAULT_PLY_WEIGHT_CLIP = (0.25, 4.0)
# Max weight multiplier at full disagreement (vs SF-best). 1.0 = style term off.
DEFAULT_STYLE_DISAGREE_BOOST = 2.0
# |delta_vs_best| in pawns that maps to strength=1 (hard cap on the ramp).
DEFAULT_STYLE_DISAGREE_SCALE = 2.0
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


# Forced-move weights (E21) — continuous in best-vs-second gap; default unused.
# delta_second_vs_best = eval(second) - eval(best) ≤ 0 (user POV among masked).
# More negative delta → more forced → weight factor → 0.
# factor = exp(delta / scale) = exp(-gap / scale), gap = best - second ≥ 0.
# Forcedness is a training weight only — do not add a forced-move eval metric slice.
DEFAULT_FORCED_SCALE_PAWNS = 1.5
_EVAL_AFTER_KEY = "evaluation_after_user_pov"


def best_vs_second_gap_pawns(
    move_feats: np.ndarray,
    mask: np.ndarray,
    *,
    tanh_scale: float = _EVAL_FEAT_TANH_SCALE,
) -> np.ndarray:
    """Per-row ``best - second_best`` in pawns (user POV eval among masked) ≥ 0.

    Rows with fewer than two legal candidates return ``0``.
    """
    return -second_vs_best_delta_pawns(move_feats, mask, tanh_scale=tanh_scale)


def second_vs_best_delta_pawns(
    move_feats: np.ndarray,
    mask: np.ndarray,
    *,
    tanh_scale: float = _EVAL_FEAT_TANH_SCALE,
) -> np.ndarray:
    """Per-row ``second_best - best`` in pawns (≤ 0). More negative = more forced.

    Rows with fewer than two legal candidates return ``0`` (no downweight).
    """
    feats = np.asarray(move_feats, dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64)
    if feats.ndim != 3:
        raise ValueError(f"move_feats expected (N, MAX, F), got {feats.shape}")
    if m.shape[:2] != feats.shape[:2]:
        raise ValueError(f"mask shape {m.shape} incompatible with feats {feats.shape}")
    try:
        eval_i = CANDIDATE_MOVE_FEAT_KEYS.index(_EVAL_AFTER_KEY)
    except ValueError as exc:
        raise ValueError(f"missing feat key {_EVAL_AFTER_KEY!r}") from exc

    packed = feats[:, :, eval_i]
    clipped = np.clip(packed, -_ATANH_CLIP, _ATANH_CLIP)
    evals = float(tanh_scale) * np.arctanh(clipped)
    legal = m > 0.5
    evals = np.where(legal, evals, -np.inf)
    ordered = np.sort(evals, axis=1)[:, ::-1]
    n_legal = np.sum(legal, axis=1)
    best = ordered[:, 0]
    second = ordered[:, 1]
    delta = second - best
    return np.where(n_legal >= 2, delta, 0.0).astype(np.float64)


def forced_move_downweight_factor(
    move_feats: np.ndarray,
    mask: np.ndarray,
    *,
    scale_pawns: float = DEFAULT_FORCED_SCALE_PAWNS,
) -> np.ndarray:
    """Continuous per-row multiplier in ``(0, 1]`` from second-vs-best delta.

    ``factor = exp(delta / scale)`` with ``delta = second - best ≤ 0``.
    Equal top-two (delta=0) → 1.0; large forced gap → approaches 0.
    """
    scale = float(scale_pawns)
    if scale <= 0:
        raise ValueError(f"scale_pawns must be > 0, got {scale}")
    delta = second_vs_best_delta_pawns(move_feats, mask)
    return np.exp(delta / scale).astype(np.float64)


def candidate_style_sample_weights(
    plies: list[int] | np.ndarray,
    move_feats: np.ndarray,
    labels: np.ndarray,
    *,
    style_disagree_boost: float | None = None,
    style_disagree_scale: float | None = None,
    lam: float = DEFAULT_PLY_WEIGHT_LAMBDA,
    clip: tuple[float, float] | None = DEFAULT_PLY_WEIGHT_CLIP,
    candidate_mask: np.ndarray | None = None,
    forced_scale_pawns: float | None = None,
) -> np.ndarray:
    """Ply weights * continuous style boost; mean-normalized (clip).

    ``style_disagree_boost`` / ``style_disagree_scale`` ``None`` -> env / defaults.
    Boost ``1.0`` disables the style term (ply only).

    Forced-move downweight (E21): pass ``candidate_mask`` and ``forced_scale_pawns``
    (>0). Multiplies by ``exp((second-best)/scale)`` continuously. Default off so
    encoder A/B stays one change family.
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
    if forced_scale_pawns is not None:
        if candidate_mask is None:
            raise ValueError("forced_scale_pawns requires candidate_mask")
        raw = raw * forced_move_downweight_factor(
            move_feats,
            candidate_mask,
            scale_pawns=float(forced_scale_pawns),
        )
    return normalize_sample_weights(raw, clip=clip)
