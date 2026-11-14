"""Unit tests for inference-blend AB helpers."""

from __future__ import annotations

import numpy as np
import pytest

from chess_teacher.pipelines.neural_network.hybrid_val_score import blend_candidate_logits
from chess_teacher.pipelines.neural_network.offline_inference_blend_ab import (
    _ARM_ALPHA,
    _ARMS,
    format_report,
)


def test_arm_alpha_map() -> None:
    assert set(_ARMS) == set(_ARM_ALPHA)
    assert _ARM_ALPHA["serve_parent"] == 0.0
    assert _ARM_ALPHA["serve_child"] == 1.0
    assert _ARM_ALPHA["blend_0.50"] == 0.5


def test_blend_candidate_logits_endpoints() -> None:
    parent = np.array([[1.0, 0.0], [0.0, 1.0]])
    child = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert np.allclose(blend_candidate_logits(parent, child, alpha=0.0), parent)
    assert np.allclose(blend_candidate_logits(parent, child, alpha=1.0), child)
    mid = blend_candidate_logits(parent, child, alpha=0.5)
    assert np.allclose(mid, np.array([[0.5, 0.5], [0.5, 0.5]]))


def test_blend_alpha_bounds() -> None:
    parent = np.zeros((1, 2))
    child = np.ones((1, 2))
    with pytest.raises(ValueError, match="alpha"):
        blend_candidate_logits(parent, child, alpha=-0.1)


def test_format_report_includes_alpha() -> None:
    report = {
        "account_id": "abc1234567890",
        "n_parent_train": 10,
        "n_user_train": 10,
        "n_user_val": 10,
        "user_style_disagree_boost": 4.0,
        "arms": {
            "serve_parent": {
                "alpha": 0.0,
                "val": {
                    "top1_overall": 0.4,
                    "top1_sf_agree": 0.7,
                    "top1_sf_disagree": 0.2,
                    "sf_delta_mean_pawns": -0.2,
                },
                "val_log_loss": 2.1,
                "fit_seconds": 1.0,
                "gate": {"reference": True},
            },
            "blend_0.50": {
                "alpha": 0.5,
                "val": {
                    "top1_overall": 0.41,
                    "top1_sf_agree": 0.71,
                    "top1_sf_disagree": 0.21,
                    "sf_delta_mean_pawns": -0.25,
                },
                "val_log_loss": 2.05,
                "fit_seconds": 1.0,
                "gate": {
                    "passes_serve_gate": True,
                    "d_top1": 0.01,
                    "d_disagree": 0.01,
                    "d_ll": -0.05,
                    "pawn_gap_regress": 0.05,
                },
            },
        },
    }
    text = format_report(report)
    assert "blend_0.50:" in text
    assert "alpha=0.50" in text
    assert "gate=PASS" in text
