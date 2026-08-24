"""Unit tests for ply + continuous SF-disagree style sample weights."""

from __future__ import annotations

import numpy as np
import pytest

from chess_teacher.pipelines.neural_network.candidate_eval import (
    CANDIDATE_MOVE_FEAT_KEYS,
    MOVE_FEAT_DIM,
)
from chess_teacher.pipelines.neural_network.ply_weights import (
    candidate_style_sample_weights,
    labeled_delta_vs_best_pawns,
    ply_sample_weights,
    ply_weight_raw,
    style_disagree_boost_from_env,
    style_disagree_scale_from_env,
    user_not_sf_best_mask,
    user_sf_disagree_strength,
)


def test_ply_weight_raw_increases_with_ply() -> None:
    early = float(ply_weight_raw(5))
    late = float(ply_weight_raw(40))
    assert late > early


def test_ply_sample_weights_mean_near_one() -> None:
    w = ply_sample_weights([1, 10, 20, 40, 60])
    assert w.dtype == np.float32
    assert float(np.mean(w)) == pytest.approx(1.0, abs=1e-5)
    assert np.min(w) >= 0.25 - 1e-6
    assert np.max(w) <= 4.0 + 1e-6


def _fake_feats_labels() -> tuple[np.ndarray, np.ndarray]:
    """Pack tanh(delta/5) for deltas 0, -1, -2, -4 pawns."""
    n = 4
    feats = np.zeros((n, 8, MOVE_FEAT_DIM), dtype=np.float32)
    labels = np.zeros((n,), dtype=np.int64)
    delta_i = CANDIDATE_MOVE_FEAT_KEYS.index("delta_vs_best")
    raw = np.asarray([0.0, -1.0, -2.0, -4.0], dtype=np.float64)
    feats[np.arange(n), 0, delta_i] = np.tanh(raw / 5.0).astype(np.float32)
    return feats, labels


def test_labeled_delta_roundtrip_approx() -> None:
    feats, labels = _fake_feats_labels()
    recovered = labeled_delta_vs_best_pawns(feats, labels)
    np.testing.assert_allclose(recovered, [0.0, -1.0, -2.0, -4.0], atol=1e-5)


def test_disagree_strength_ramp_with_scale_2() -> None:
    feats, labels = _fake_feats_labels()
    # scale=2: 0 -> 0, 1p -> 0.5, 2p -> 1, 4p -> 1 (capped)
    s = user_sf_disagree_strength(feats, labels, scale_pawns=2.0)
    np.testing.assert_allclose(s, [0.0, 0.5, 1.0, 1.0], atol=1e-5)


def test_user_not_sf_best_mask() -> None:
    feats, labels = _fake_feats_labels()
    mask = user_not_sf_best_mask(feats, labels)
    assert mask.tolist() == [False, True, True, True]


def test_style_boost_1_matches_ply_only() -> None:
    feats, labels = _fake_feats_labels()
    plies = [10, 10, 10, 10]
    w_ply = ply_sample_weights(plies)
    w = candidate_style_sample_weights(
        plies,
        feats,
        labels,
        style_disagree_boost=1.0,
        style_disagree_scale=2.0,
    )
    np.testing.assert_allclose(w, w_ply, rtol=1e-5)


def test_continuous_boost_orders_by_strength() -> None:
    feats, labels = _fake_feats_labels()
    plies = [10, 10, 10, 10]
    w = candidate_style_sample_weights(
        plies,
        feats,
        labels,
        style_disagree_boost=3.0,
        style_disagree_scale=2.0,
    )
    # After mean-norm: deeper disagreement (rows 2,3) >= mild (row 1) > agree (0)
    assert float(w[1]) > float(w[0])
    assert float(w[2]) >= float(w[1]) - 1e-6
    assert float(w[3]) == pytest.approx(float(w[2]), abs=1e-5)  # both at cap
    assert float(np.mean(w)) == pytest.approx(1.0, abs=1e-5)


def test_style_knobs_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BASELINE_STYLE_DISAGREE_BOOST", raising=False)
    monkeypatch.delenv("BASELINE_STYLE_DISAGREE_SCALE", raising=False)
    assert style_disagree_boost_from_env(default=2.0) == pytest.approx(2.0)
    assert style_disagree_scale_from_env(default=2.0) == pytest.approx(2.0)
    monkeypatch.setenv("BASELINE_STYLE_DISAGREE_BOOST", "1.5")
    monkeypatch.setenv("BASELINE_STYLE_DISAGREE_SCALE", "2.5")
    assert style_disagree_boost_from_env() == pytest.approx(1.5)
    assert style_disagree_scale_from_env() == pytest.approx(2.5)
    monkeypatch.setenv("BASELINE_STYLE_DISAGREE_SCALE", "0")
    assert style_disagree_scale_from_env(default=2.0) == pytest.approx(2.0)
