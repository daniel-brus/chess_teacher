"""Unit tests for personal serve/promote gate."""

from __future__ import annotations

from chess_teacher.pipelines.neural_network.eval_metrics import EvalMetrics
from chess_teacher.pipelines.neural_network.ply_weights import DEFAULT_USER_STYLE_DISAGREE_BOOST
from chess_teacher.pipelines.neural_network.serve_gate import (
    evaluate_serve_gate,
    evaluate_serve_gate_metrics,
    serve_gate_reason,
)


def test_user_style_boost_default_is_option_b() -> None:
    assert DEFAULT_USER_STYLE_DISAGREE_BOOST == 2.0


def test_evaluate_serve_gate_slice_d_boost2_passes() -> None:
    gate = evaluate_serve_gate(
        parent_top1=0.385,
        parent_disagree=0.177,
        parent_ll=2.212,
        parent_pawn_gap=-0.20,
        child_top1=0.377,
        child_disagree=0.189,
        child_ll=2.227,
        child_pawn_gap=-0.38,
    )
    assert gate["passes_serve_gate"] is True


def test_evaluate_serve_gate_metrics_missing_disagree() -> None:
    parent = EvalMetrics(
        top1_overall=0.4,
        top3_overall=0.5,
        top1_sf_agree=None,
        top3_sf_agree=None,
        top1_sf_disagree=None,
        top3_sf_disagree=None,
        top1_overall_weighted=0.4,
        n_eval=1,
        n_dropped=0,
        n_sf_agree=0,
        n_sf_disagree=0,
        sf_disagree_frac=0.0,
    )
    child = parent
    gate = evaluate_serve_gate_metrics(parent, child)
    assert gate["passes_serve_gate"] is False
    assert "missing" in serve_gate_reason(gate)
