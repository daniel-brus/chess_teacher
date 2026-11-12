"""Unit tests for serve-gate AB helpers."""

from __future__ import annotations

import pytest

from chess_teacher.pipelines.neural_network.offline_serve_gate_ab import (
    _ARM_BOOST,
    _ARMS,
    evaluate_serve_gate,
    format_report,
)


def test_arm_boost_map_covers_all_arms() -> None:
    assert set(_ARMS) == set(_ARM_BOOST)
    assert _ARM_BOOST["serve_parent"] is None
    assert _ARM_BOOST["ft_boost_1"] == 1.0
    assert _ARM_BOOST["ft_boost_2"] == 2.0
    assert _ARM_BOOST["ft_boost_4"] == 4.0


def test_evaluate_serve_gate_passes_when_trade_is_good() -> None:
    gate = evaluate_serve_gate(
        parent_top1=0.40,
        parent_disagree=0.20,
        parent_ll=2.10,
        parent_pawn_gap=-0.18,
        child_top1=0.395,
        child_disagree=0.22,
        child_ll=2.11,
        child_pawn_gap=-0.30,
    )
    assert gate["passes_serve_gate"] is True
    assert gate["pass_disagree"] is True
    assert gate["pass_overall"] is True
    assert gate["pass_pawn"] is True
    assert gate["ll_warn"] is False


def test_evaluate_serve_gate_fails_on_pawn_or_overall() -> None:
    # Slice C-like: disagree lift OK, overall + pawn blow up.
    gate = evaluate_serve_gate(
        parent_top1=0.413,
        parent_disagree=0.200,
        parent_ll=2.096,
        parent_pawn_gap=-0.18,
        child_top1=0.399,
        child_disagree=0.228,
        child_ll=2.102,
        child_pawn_gap=-0.59,
    )
    assert gate["pass_disagree"] is True
    assert gate["pass_overall"] is False  # -1.4 pts > 1.0 ceiling
    assert gate["pass_pawn"] is False  # regress 0.41 > 0.20
    assert gate["passes_serve_gate"] is False


def test_format_report_marks_gate() -> None:
    report = {
        "account_id": "abc1234567890",
        "n_parent_train": 100,
        "n_user_train": 50,
        "n_user_val": 20,
        "epochs_warm": 2,
        "serve_gate": {
            "min_disagree_lift": 0.01,
            "max_overall_drop": 0.01,
            "max_pawn_gap_regress": 0.20,
            "ll_warn": 0.03,
        },
        "arms": {
            "serve_parent": {
                "style_disagree_boost": None,
                "val": {
                    "top1_overall": 0.41,
                    "top1_sf_agree": 0.78,
                    "top1_sf_disagree": 0.20,
                    "sf_delta_mean_pawns": -0.18,
                },
                "val_log_loss": 2.10,
                "fit_seconds": 1.0,
                "gate": {"reference": True, "passes_serve_gate": False},
            },
            "ft_boost_4": {
                "style_disagree_boost": 4.0,
                "val": {
                    "top1_overall": 0.40,
                    "top1_sf_agree": 0.70,
                    "top1_sf_disagree": 0.23,
                    "sf_delta_mean_pawns": -0.30,
                },
                "val_log_loss": 2.11,
                "fit_seconds": 2.0,
                "gate": {
                    "passes_serve_gate": True,
                    "d_top1": -0.01,
                    "d_disagree": 0.03,
                    "d_ll": 0.01,
                    "pawn_gap_regress": 0.12,
                    "ll_warn": False,
                },
            },
        },
    }
    text = format_report(report)
    assert "serve_parent:" in text
    assert "ft_boost_4:" in text
    assert "gate=PASS" in text


def test_run_requires_serve_parent_arm() -> None:
    from chess_teacher.pipelines.neural_network.offline_serve_gate_ab import (
        run_serve_gate_ab,
    )

    with pytest.raises(ValueError, match="serve_parent"):
        run_serve_gate_ab(
            account_id="x",
            parent_train_limit=10,
            user_train_limit=10,
            val_limit=10,
            parent_epochs=1,
            epochs=1,
            arms=["ft_boost_4"],
            batch_size=64,
            parent_style_disagree_boost=2.0,
            style_disagree_scale=2.0,
            split_version="baseline-v1",
            output=__import__("pathlib").Path("storage/tmp/_unused.json"),
        )
