"""Unit tests for personal-head AB helpers."""

from __future__ import annotations

from chess_teacher.pipelines.neural_network.offline_personal_head_ab import (
    _ARM_FREEZE,
    _ARMS,
    format_report,
)


def test_arm_freeze_map_covers_all_arms() -> None:
    assert set(_ARMS) == set(_ARM_FREEZE)
    assert _ARM_FREEZE["full_ft"] is False
    assert _ARM_FREEZE["freeze_trunk"] is True


def test_format_report_delta_vs_full_ft() -> None:
    report = {
        "account_id": "abc1234567890",
        "n_parent_train": 100,
        "n_user_train": 50,
        "n_user_val": 20,
        "epochs_warm": 2,
        "user_style_disagree_boost": 4.0,
        "parent": {
            "val": {
                "top1_overall": 0.40,
                "top1_sf_disagree": 0.20,
                "sf_delta_mean_pawns": -0.1,
            },
            "val_log_loss": 2.1,
        },
        "arms": {
            "full_ft": {
                "freeze_board_trunk": False,
                "val": {
                    "top1_overall": 0.42,
                    "top1_sf_agree": 0.70,
                    "top1_sf_disagree": 0.22,
                    "sf_delta_mean_pawns": -0.12,
                },
                "val_log_loss": 2.0,
                "fit_seconds": 1.0,
            },
            "freeze_trunk": {
                "freeze_board_trunk": True,
                "val": {
                    "top1_overall": 0.43,
                    "top1_sf_agree": 0.71,
                    "top1_sf_disagree": 0.23,
                    "sf_delta_mean_pawns": -0.11,
                },
                "val_log_loss": 1.95,
                "fit_seconds": 0.8,
            },
        },
    }
    text = format_report(report)
    assert "full_ft:" in text
    assert "freeze_trunk:" in text
    assert "d_top1=+0.0100" in text
    assert "d_ll=-0.0500" in text
