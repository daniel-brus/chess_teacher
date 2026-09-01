"""Unit tests for stratified candidate-style eval metrics."""

from __future__ import annotations

import numpy as np

from chess_teacher.pipelines.neural_network.candidate_eval import (
    CANDIDATE_MOVE_FEAT_KEYS,
    MOVE_FEAT_DIM,
)
from chess_teacher.pipelines.neural_network.eval_metrics import compute_candidate_style_metrics


def _synthetic_batch(
    *,
    n: int = 4,
    max_candidates: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Perfect top-1 predictions with mixed SF agree/disagree labels."""
    feats = np.zeros((n, max_candidates, MOVE_FEAT_DIM), dtype=np.float32)
    mask = np.ones((n, max_candidates), dtype=np.float32)
    labels = np.arange(n, dtype=np.int64) % max_candidates
    logits = np.full((n, max_candidates), -10.0, dtype=np.float64)
    for i in range(n):
        logits[i, labels[i]] = 10.0
    delta_i = CANDIDATE_MOVE_FEAT_KEYS.index("delta_vs_best")
    # rows 0,2 agree (delta 0); rows 1,3 disagree
    deltas = [0.0, -1.0, 0.0, -2.0]
    for i, delta in enumerate(deltas[:n]):
        feats[i, labels[i], delta_i] = np.tanh(delta / 5.0)
    plies = [10] * n
    return logits, mask, labels, feats, plies


def test_perfect_predictions_top1_is_one() -> None:
    logits, mask, labels, feats, plies = _synthetic_batch()
    m = compute_candidate_style_metrics(
        logits=logits,
        mask=mask,
        labels=labels,
        move_feats=feats,
        plies=plies,
        n_input=len(labels),
        max_candidates=8,
    )
    assert m.top1_overall == 1.0
    assert m.top3_overall == 1.0
    assert m.top1_sf_agree == 1.0
    assert m.top1_sf_disagree == 1.0
    assert m.n_sf_agree == 2
    assert m.n_sf_disagree == 2


def test_wrong_top1_on_disagree_only() -> None:
    logits, mask, labels, feats, plies = _synthetic_batch(n=4)
    # Flip prediction on disagree rows only (1 and 3)
    for row in (1, 3):
        wrong = (labels[row] + 1) % mask.shape[1]
        logits[row, labels[row]] = -10.0
        logits[row, wrong] = 10.0
    m = compute_candidate_style_metrics(
        logits=logits,
        mask=mask,
        labels=labels,
        move_feats=feats,
        plies=plies,
        n_input=4,
        max_candidates=8,
    )
    assert m.top1_overall == 0.5
    assert m.top1_sf_agree == 1.0
    assert m.top1_sf_disagree == 0.0


def test_as_dict_includes_stratified_keys() -> None:
    logits, mask, labels, feats, plies = _synthetic_batch(n=2)
    m = compute_candidate_style_metrics(
        logits=logits,
        mask=mask,
        labels=labels,
        move_feats=feats,
        plies=plies,
        n_input=2,
        max_candidates=8,
    )
    d = m.as_dict()
    assert "top1_sf_agree" in d
    assert "top1_sf_disagree" in d
    assert d["n_eval"] == 2.0
