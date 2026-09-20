"""Unit tests for ply + continuous SF-disagree style sample weights."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from chess_teacher.pipelines.neural_network.candidate_eval import (
    CANDIDATE_MOVE_FEAT_KEYS,
    MOVE_FEAT_DIM,
)
from chess_teacher.pipelines.neural_network.ply_weights import (
    baseline_disagree_strength,
    candidate_style_sample_weights,
    labeled_delta_vs_best_pawns,
    ply_sample_weights,
    ply_weight_raw,
    recency_weight_raw,
    style_disagree_boost_from_env,
    style_disagree_scale_from_env,
    user_finetune_sample_weights,
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


def test_recency_weight_raw_upweights_newer() -> None:
    now = datetime(2026, 6, 1, tzinfo=UTC)
    older = datetime(2026, 1, 1, tzinfo=UTC)
    newer = datetime(2026, 5, 1, tzinfo=UTC)
    w = recency_weight_raw([older, newer], lam=0.02, now=now)
    assert float(w[1]) > float(w[0])
    assert float(w[0]) > 0.0


def test_recency_weight_raw_lam_zero_is_ones() -> None:
    now = datetime(2026, 6, 1, tzinfo=UTC)
    older = datetime(2026, 1, 1, tzinfo=UTC)
    newer = datetime(2026, 5, 1, tzinfo=UTC)
    w = recency_weight_raw([older, newer], lam=0.0, now=now)
    np.testing.assert_allclose(w, [1.0, 1.0])


def test_recency_weight_raw_naive_datetime_treated_as_utc() -> None:
    now = datetime(2026, 6, 1, tzinfo=UTC)
    naive = datetime(2026, 5, 1)
    aware = datetime(2026, 5, 1, tzinfo=UTC)
    w = recency_weight_raw([naive, aware], lam=0.02, now=now)
    np.testing.assert_allclose(w[0], w[1])


def test_user_finetune_sample_weights_mean_near_one() -> None:
    feats, labels = _fake_feats_labels()
    plies = [10, 10, 10, 10]
    now = datetime(2026, 6, 1, tzinfo=UTC)
    end_times = [
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 3, 1, tzinfo=UTC),
        datetime(2026, 5, 1, tzinfo=UTC),
        datetime(2026, 5, 15, tzinfo=UTC),
    ]
    w = user_finetune_sample_weights(
        plies,
        feats,
        labels,
        end_times,
        recency_lambda=0.02,
        style_disagree_boost=2.0,
        style_disagree_scale=2.0,
        now=now,
    )
    assert w.dtype == np.float32
    assert float(np.mean(w)) == pytest.approx(1.0, abs=1e-5)


def test_user_finetune_lam_zero_matches_candidate_style() -> None:
    feats, labels = _fake_feats_labels()
    plies = [10, 20, 30, 40]
    now = datetime(2026, 6, 1, tzinfo=UTC)
    end_times = [
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 2, 1, tzinfo=UTC),
        datetime(2026, 3, 1, tzinfo=UTC),
        datetime(2026, 4, 1, tzinfo=UTC),
    ]
    w_user = user_finetune_sample_weights(
        plies,
        feats,
        labels,
        end_times,
        recency_lambda=0.0,
        style_disagree_boost=3.0,
        style_disagree_scale=2.0,
        now=now,
    )
    w_style = candidate_style_sample_weights(
        plies,
        feats,
        labels,
        style_disagree_boost=3.0,
        style_disagree_scale=2.0,
    )
    np.testing.assert_allclose(w_user, w_style, rtol=1e-5)


def test_user_finetune_recency_survives_ply_style_clip() -> None:
    feats, labels = _fake_feats_labels()
    # Same ply and same style (agree / boost off) so only recency should order.
    plies = [10, 10, 10, 10]
    now = datetime(2026, 6, 1, tzinfo=UTC)
    end_times = [
        datetime(2025, 6, 1, tzinfo=UTC),
        datetime(2025, 6, 1, tzinfo=UTC),
        datetime(2026, 5, 31, tzinfo=UTC),
        datetime(2026, 5, 31, tzinfo=UTC),
    ]
    w = user_finetune_sample_weights(
        plies,
        feats,
        labels,
        end_times,
        recency_lambda=0.02,
        style_disagree_boost=1.0,
        style_disagree_scale=2.0,
        now=now,
    )
    assert float(w[2]) > float(w[0])
    assert float(w[3]) > float(w[1])


def test_user_finetune_recency_boost_one_is_off() -> None:
    feats, labels = _fake_feats_labels()
    plies = [10, 20, 30, 40]
    now = datetime(2026, 6, 1, tzinfo=UTC)
    end_times = [
        datetime(2025, 6, 1, tzinfo=UTC),
        datetime(2025, 9, 1, tzinfo=UTC),
        datetime(2026, 3, 1, tzinfo=UTC),
        datetime(2026, 5, 31, tzinfo=UTC),
    ]
    w_user = user_finetune_sample_weights(
        plies,
        feats,
        labels,
        end_times,
        recency_lambda=0.02,
        recency_boost=1.0,
        style_disagree_boost=3.0,
        style_disagree_scale=2.0,
        now=now,
    )
    w_style = candidate_style_sample_weights(
        plies,
        feats,
        labels,
        style_disagree_boost=3.0,
        style_disagree_scale=2.0,
    )
    np.testing.assert_allclose(w_user, w_style, rtol=1e-5)


def test_user_finetune_newest_weight_gt_old_but_capped() -> None:
    feats, labels = _fake_feats_labels()
    plies = [10, 10, 10, 10]
    now = datetime(2026, 6, 1, tzinfo=UTC)
    end_times = [
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2026, 6, 1, tzinfo=UTC),
        datetime(2026, 6, 1, tzinfo=UTC),
    ]
    boost = 2.0
    w = user_finetune_sample_weights(
        plies,
        feats,
        labels,
        end_times,
        recency_lambda=0.02,
        recency_boost=boost,
        style_disagree_boost=1.0,
        style_disagree_scale=2.0,
        now=now,
    )
    assert float(w[2]) > float(w[0])
    assert float(w[3]) > float(w[1])
    ratio = float(w[2] / w[0])
    assert ratio <= boost + 1e-5


def test_user_finetune_missing_end_time_not_newest() -> None:
    feats, labels = _fake_feats_labels()
    plies = [10, 10, 10, 10]
    now = datetime(2026, 6, 1, tzinfo=UTC)
    end_times = [
        datetime(2024, 1, 1, tzinfo=UTC),
        None,
        datetime(2026, 6, 1, tzinfo=UTC),
        datetime(2026, 6, 1, tzinfo=UTC),
    ]
    w = user_finetune_sample_weights(
        plies,
        feats,
        labels,
        end_times,
        recency_lambda=0.02,
        recency_boost=2.0,
        style_disagree_boost=1.0,
        style_disagree_scale=2.0,
        now=now,
    )
    assert float(w[1]) < float(w[2])
    assert float(w[1]) == pytest.approx(float(w[0]), rel=1e-3, abs=1e-3)


def test_recency_weight_raw_missing_end_time_is_zero() -> None:
    now = datetime(2026, 6, 1, tzinfo=UTC)
    newer = datetime(2026, 5, 1, tzinfo=UTC)
    w = recency_weight_raw([None, newer], lam=0.02, now=now)
    assert float(w[0]) == 0.0
    assert float(w[1]) > float(w[0])


def test_baseline_disagree_hard_mask() -> None:
    n, max_c = 3, 4
    logits = np.zeros((n, max_c), dtype=np.float64)
    mask = np.ones((n, max_c), dtype=np.float64)
    logits[0, 0] = 5.0
    logits[1, 1] = 5.0
    logits[2, 2] = 9.0
    mask[2, 2] = 0.0
    logits[2, 0] = 1.0
    labels = np.asarray([0, 0, 0], dtype=np.int64)
    strength = baseline_disagree_strength(logits, mask, labels)
    np.testing.assert_allclose(strength, [0.0, 1.0, 0.0])


def test_user_finetune_baseline_mask_boosts_disagree() -> None:
    feats, labels = _fake_feats_labels()
    plies = [10, 10, 10, 10]
    now = datetime(2026, 6, 1, tzinfo=UTC)
    end_times = [datetime(2026, 5, 1, tzinfo=UTC)] * 4
    baseline_s = np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float64)
    w = user_finetune_sample_weights(
        plies,
        feats,
        labels,
        end_times,
        recency_lambda=0.0,
        recency_boost=1.0,
        style_disagree_boost=1.0,
        style_disagree_scale=2.0,
        baseline_disagree_strength=baseline_s,
        baseline_disagree_boost=4.0,
        now=now,
    )
    assert float(w[1]) > float(w[0])
    assert float(w[2]) > float(w[3])
    assert float(np.mean(w)) == pytest.approx(1.0, abs=1e-5)
