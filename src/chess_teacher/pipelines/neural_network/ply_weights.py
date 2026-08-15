"""Sample weights that down-weight early plies (opening-heavy dataset).

Simple exponential: ``w(ply) ∝ exp(λ · ply)``, then mean-normalized over the
batch (or provided sample). Optional clip keeps rare late plies from exploding.

λ ≈ 0.016 fitted loosely to local ``games.moves`` ply histogram (exp count model);
not a precision fit — intentional.
"""

from __future__ import annotations

import numpy as np

DEFAULT_PLY_WEIGHT_LAMBDA = 0.016
DEFAULT_PLY_WEIGHT_CLIP = (0.25, 4.0)


def ply_weight_raw(ply: int | np.ndarray, *, lam: float = DEFAULT_PLY_WEIGHT_LAMBDA) -> np.ndarray:
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
