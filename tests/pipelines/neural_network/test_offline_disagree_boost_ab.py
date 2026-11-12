"""Unit tests for disagree-boost A/B formatting helpers."""

from __future__ import annotations

from chess_teacher.pipelines.neural_network.offline_disagree_boost_ab import (
    _ARM_BOOST,
    format_report,
)


def test_arm_boost_map() -> None:
    assert _ARM_BOOST["boost_2"] == 2.0
    assert _ARM_BOOST["boost_1"] == 1.0


def test_format_report_includes_deltas() -> None:
    report = {
        "n_train_loaded": 100,
        "n_val_loaded": 50,
        "epochs": 2,
        "style_disagree_scale": 2.0,
        "arms": {
            "boost_2": {
                "style_disagree_boost": 2.0,
                "val_log_loss": 2.0,
                "fit_seconds": 10.0,
                "val": {
                    "top1_overall": 0.40,
                    "top1_sf_agree": 0.70,
                    "top1_sf_disagree": 0.20,
                    "sf_delta_mean_pawns": -0.1,
                },
            },
            "boost_1": {
                "style_disagree_boost": 1.0,
                "val_log_loss": 1.9,
                "fit_seconds": 11.0,
                "val": {
                    "top1_overall": 0.41,
                    "top1_sf_agree": 0.71,
                    "top1_sf_disagree": 0.21,
                    "sf_delta_mean_pawns": -0.1,
                },
            },
        },
    }
    text = format_report(report)
    assert "boost_2:" in text
    assert "boost_1:" in text
    assert "d_top1=+0.0100" in text
    assert "d_ll=-0.1000" in text
