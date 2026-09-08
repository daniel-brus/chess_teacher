"""Tests for continuous forced-move weight helpers (E21)."""

from __future__ import annotations

import numpy as np
import pytest

from chess_teacher.pipelines.neural_network.candidate_eval import (
    CANDIDATE_MOVE_FEAT_KEYS,
    MOVE_FEAT_DIM,
)
from chess_teacher.pipelines.neural_network.ply_weights import (
    best_vs_second_gap_pawns,
    candidate_style_sample_weights,
    forced_move_downweight_factor,
    second_vs_best_delta_pawns,
)


def _feats_with_evals(evals_pawns: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    n = len(evals_pawns)
    max_c = max(len(row) for row in evals_pawns)
    feats = np.zeros((n, max_c, MOVE_FEAT_DIM), dtype=np.float32)
    mask = np.zeros((n, max_c), dtype=np.float32)
    eval_i = CANDIDATE_MOVE_FEAT_KEYS.index("evaluation_after_user_pov")
    for i, row in enumerate(evals_pawns):
        for j, ev in enumerate(row):
            feats[i, j, eval_i] = np.tanh(ev / 5.0)
            mask[i, j] = 1.0
    return feats, mask


def test_second_vs_best_delta_and_gap() -> None:
    feats, mask = _feats_with_evals(
        [
            [3.0, 1.0, 0.0],  # delta -2.0
            [0.5, 0.4],  # delta -0.1
            [1.0],  # single -> 0
        ]
    )
    delta = second_vs_best_delta_pawns(feats, mask)
    gap = best_vs_second_gap_pawns(feats, mask)
    np.testing.assert_allclose(delta, [-2.0, -0.1, 0.0], atol=1e-4)
    np.testing.assert_allclose(gap, -delta, atol=1e-4)


def test_continuous_downweight_monotone_in_gap() -> None:
    feats, mask = _feats_with_evals([[3.0, 1.0], [0.5, 0.4], [1.0, 1.0]])
    factor = forced_move_downweight_factor(feats, mask, scale_pawns=1.5)
    # gap 2.0 -> exp(-2/1.5); gap 0.1 -> exp(-0.1/1.5); gap 0 -> 1
    expected = np.exp(np.asarray([-2.0, -0.1, 0.0]) / 1.5)
    np.testing.assert_allclose(factor, expected, atol=1e-4)
    assert factor[0] < factor[1] < factor[2]
    assert factor[2] == pytest.approx(1.0)


def test_sample_weights_forced_continuous() -> None:
    feats, mask = _feats_with_evals([[3.0, 1.0], [0.5, 0.4]])
    labels = np.zeros(2, dtype=np.int64)
    delta_i = CANDIDATE_MOVE_FEAT_KEYS.index("delta_vs_best")
    feats[:, 0, delta_i] = 0.0
    w_off = candidate_style_sample_weights(
        [10, 10],
        feats,
        labels,
        style_disagree_boost=1.0,
    )
    w_on = candidate_style_sample_weights(
        [10, 10],
        feats,
        labels,
        style_disagree_boost=1.0,
        candidate_mask=mask,
        forced_scale_pawns=1.5,
    )
    # Larger gap row should shrink relative to free row after normalize.
    assert float(w_on[0] / w_on[1]) < float(w_off[0] / w_off[1])


def test_forced_scale_requires_mask() -> None:
    feats, _mask = _feats_with_evals([[3.0, 1.0]])
    labels = np.zeros(1, dtype=np.int64)
    with pytest.raises(ValueError, match="candidate_mask"):
        candidate_style_sample_weights(
            [10],
            feats,
            labels,
            forced_scale_pawns=1.5,
        )
