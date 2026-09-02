"""Stratified candidate-style eval metrics for offline experiments.

Reports overall top-1 / top-3 plus SF-agree and SF-disagree subsets.
See ``.agents/docs/ml-training-roadmap.md`` Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from chess_teacher.pipelines.neural_network.candidate_eval import MAX_CANDIDATES
from chess_teacher.pipelines.neural_network.create_training_set import (
    TrainingBatch,
    TrainingDatum,
)
from chess_teacher.pipelines.neural_network.ply_weights import (
    candidate_style_sample_weights,
    user_not_sf_best_mask,
)


@dataclass(frozen=True)
class EvalMetrics:
    """Candidate-style top-k metrics on one eval set."""

    top1_overall: float
    top3_overall: float
    top1_sf_agree: float | None
    top3_sf_agree: float | None
    top1_sf_disagree: float | None
    top3_sf_disagree: float | None
    top1_overall_weighted: float
    n_eval: int
    n_dropped: int
    n_sf_agree: int
    n_sf_disagree: int
    sf_disagree_frac: float

    def as_dict(self) -> dict[str, float]:
        out: dict[str, float] = {
            "top1_overall": self.top1_overall,
            "top3_overall": self.top3_overall,
            "top1_overall_weighted": self.top1_overall_weighted,
            "n_eval": float(self.n_eval),
            "n_dropped": float(self.n_dropped),
            "n_sf_agree": float(self.n_sf_agree),
            "n_sf_disagree": float(self.n_sf_disagree),
            "sf_disagree_frac": self.sf_disagree_frac,
        }
        if self.top1_sf_agree is not None:
            out["top1_sf_agree"] = self.top1_sf_agree
        if self.top3_sf_agree is not None:
            out["top3_sf_agree"] = self.top3_sf_agree
        if self.top1_sf_disagree is not None:
            out["top1_sf_disagree"] = self.top1_sf_disagree
        if self.top3_sf_disagree is not None:
            out["top3_sf_disagree"] = self.top3_sf_disagree
        return out


def _topk_hits(
    logits: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    *,
    max_candidates: int,
) -> tuple[np.ndarray, np.ndarray]:
    masked = np.where(mask > 0.5, logits, -np.inf)
    y_index = np.asarray(labels, dtype=np.int64)
    top1 = np.argmax(masked, axis=1) == y_index
    k3 = min(3, max_candidates)
    part3 = np.argpartition(masked, -k3, axis=1)[:, -k3:]
    top3 = np.any(part3 == y_index.reshape(-1, 1), axis=1)
    return top1, top3


def _mean_or_none(hits: np.ndarray, selector: np.ndarray) -> float | None:
    idx = np.flatnonzero(selector)
    if idx.size == 0:
        return None
    return float(np.mean(hits[idx]))


def compute_candidate_style_metrics(
    *,
    logits: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    move_feats: np.ndarray,
    plies: list[int] | np.ndarray,
    n_input: int,
    max_candidates: int = MAX_CANDIDATES,
) -> EvalMetrics:
    """Score packed candidate tensors (no model — useful for unit tests)."""
    logits_arr = np.asarray(logits, dtype=np.float64)
    if logits_arr.ndim != 2 or logits_arr.shape[1] != max_candidates:
        raise ValueError(f"Unexpected logits shape {logits_arr.shape}")

    top1, top3 = _topk_hits(logits_arr, mask, labels, max_candidates=max_candidates)
    disagree = user_not_sf_best_mask(move_feats, labels)
    agree = ~disagree

    weights = candidate_style_sample_weights(plies, move_feats, labels)
    w_sum = float(np.sum(weights))
    top1_weighted = (
        float(np.sum(top1.astype(np.float64) * weights) / w_sum) if w_sum else 0.0
    )

    return EvalMetrics(
        top1_overall=float(np.mean(top1)),
        top3_overall=float(np.mean(top3)),
        top1_sf_agree=_mean_or_none(top1, agree),
        top3_sf_agree=_mean_or_none(top3, agree),
        top1_sf_disagree=_mean_or_none(top1, disagree),
        top3_sf_disagree=_mean_or_none(top3, disagree),
        top1_overall_weighted=top1_weighted,
        n_eval=len(labels),
        n_dropped=int(n_input) - len(labels),
        n_sf_agree=int(np.sum(agree)),
        n_sf_disagree=int(np.sum(disagree)),
        sf_disagree_frac=float(np.mean(disagree)),
    )


def evaluate_datums(
    model: Any,
    datums: list[TrainingDatum],
    *,
    max_candidates: int = MAX_CANDIDATES,
) -> EvalMetrics:
    """Run ``model.predict`` on datums and return stratified metrics."""
    if not datums:
        raise ValueError("evaluate_datums requires a non-empty datum list")

    batch = TrainingBatch(datums)
    feats, mask, labels, kept = batch.candidate_style_targets()
    if not kept:
        raise ValueError(
            "evaluate_datums: no datums with usable candidate_evaluations "
            "(user move must be in evals)"
        )
    kept_datums = [datums[i] for i in kept]
    x_state = TrainingBatch(kept_datums).state_matrix()
    logits = np.asarray(
        model.predict({"state": x_state, "move_feats": feats}, verbose=0),
        dtype=np.float64,
    )
    return compute_candidate_style_metrics(
        logits=logits,
        mask=mask,
        labels=labels,
        move_feats=feats,
        plies=[d.ply for d in kept_datums],
        n_input=len(datums),
        max_candidates=max_candidates,
    )


def format_eval_metrics(name: str, metrics: EvalMetrics) -> str:
    """Single-line human-readable summary for scripts."""
    agree_t1 = (
        f"{metrics.top1_sf_agree:.4f}" if metrics.top1_sf_agree is not None else "n/a"
    )
    disagree_t1 = (
        f"{metrics.top1_sf_disagree:.4f}"
        if metrics.top1_sf_disagree is not None
        else "n/a"
    )
    return (
        f"{name:5s} top1={metrics.top1_overall:.4f} top3={metrics.top3_overall:.4f} "
        f"agree_t1={agree_t1} disagree_t1={disagree_t1} "
        f"n={metrics.n_eval} dropped={metrics.n_dropped} "
        f"disagree_frac={metrics.sf_disagree_frac:.3f}"
    )
