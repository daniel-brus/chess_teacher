"""Draft serve/promote gate for personal children vs parent (Phase 5 Option B).

Serve / promote the child only when all hard checks hold vs parent on that
user's registry-val:

1. Disagree top-1 lift ≥ +1.0 pt
2. Overall top-1 drop ≤ 1.0 pt
3. Pawn-gap regress ≤ 0.20 pawns

Log-loss Δ > 0.03 is a soft warn only (not a hard fail).
"""

from __future__ import annotations

from typing import Any

from chess_teacher.pipelines.neural_network.eval_metrics import EvalMetrics

SERVE_GATE_MIN_DISAGREE_LIFT = 0.01
SERVE_GATE_MAX_OVERALL_DROP = 0.01
SERVE_GATE_MAX_PAWN_GAP_REGRESS = 0.20
SERVE_GATE_LL_WARN = 0.03


def evaluate_serve_gate(
    *,
    parent_top1: float,
    parent_disagree: float,
    parent_ll: float,
    parent_pawn_gap: float,
    child_top1: float,
    child_disagree: float,
    child_ll: float,
    child_pawn_gap: float,
    min_disagree_lift: float = SERVE_GATE_MIN_DISAGREE_LIFT,
    max_overall_drop: float = SERVE_GATE_MAX_OVERALL_DROP,
    max_pawn_gap_regress: float = SERVE_GATE_MAX_PAWN_GAP_REGRESS,
    ll_warn: float = SERVE_GATE_LL_WARN,
) -> dict[str, Any]:
    """Score child vs parent under the draft serve/promote gate."""
    d_top1 = float(child_top1) - float(parent_top1)
    d_disagree = float(child_disagree) - float(parent_disagree)
    d_ll = float(child_ll) - float(parent_ll)
    d_pawn = float(child_pawn_gap) - float(parent_pawn_gap)
    # More negative pawn gap = worse; regress = how much worse vs parent.
    pawn_regress = float(parent_pawn_gap) - float(child_pawn_gap)
    # Tiny epsilon so exact threshold values (e.g. +0.01) survive float noise.
    eps = 1e-12
    pass_disagree = d_disagree + eps >= float(min_disagree_lift)
    pass_overall = d_top1 + eps >= -float(max_overall_drop)
    pass_pawn = pawn_regress <= float(max_pawn_gap_regress) + eps
    ll_warn_flag = d_ll > float(ll_warn)
    passes = bool(pass_disagree and pass_overall and pass_pawn)
    return {
        "d_top1": d_top1,
        "d_disagree": d_disagree,
        "d_ll": d_ll,
        "d_pawn_gap": d_pawn,
        "pawn_gap_regress": pawn_regress,
        "pass_disagree": pass_disagree,
        "pass_overall": pass_overall,
        "pass_pawn": pass_pawn,
        "ll_warn": ll_warn_flag,
        "passes_serve_gate": passes,
    }


def evaluate_serve_gate_metrics(
    parent: EvalMetrics,
    child: EvalMetrics,
    *,
    parent_ll: float | None = None,
    child_ll: float | None = None,
) -> dict[str, Any]:
    """Serve gate from stratified ``EvalMetrics`` (LL optional; missing → 0 Δ)."""
    parent_dis = parent.top1_sf_disagree
    child_dis = child.top1_sf_disagree
    if parent_dis is None or child_dis is None:
        return {
            "d_top1": float(child.top1_overall) - float(parent.top1_overall),
            "d_disagree": float("nan"),
            "d_ll": 0.0,
            "d_pawn_gap": float(child.sf_delta_mean_pawns) - float(parent.sf_delta_mean_pawns),
            "pawn_gap_regress": float(parent.sf_delta_mean_pawns)
            - float(child.sf_delta_mean_pawns),
            "pass_disagree": False,
            "pass_overall": False,
            "pass_pawn": False,
            "ll_warn": False,
            "passes_serve_gate": False,
            "missing_disagree": True,
        }
    p_ll = 0.0 if parent_ll is None else float(parent_ll)
    c_ll = 0.0 if child_ll is None else float(child_ll)
    # When LL unknown, use equal values so soft warn stays off.
    if parent_ll is None or child_ll is None:
        p_ll, c_ll = 0.0, 0.0
    return evaluate_serve_gate(
        parent_top1=float(parent.top1_overall),
        parent_disagree=float(parent_dis),
        parent_ll=p_ll,
        parent_pawn_gap=float(parent.sf_delta_mean_pawns),
        child_top1=float(child.top1_overall),
        child_disagree=float(child_dis),
        child_ll=c_ll,
        child_pawn_gap=float(child.sf_delta_mean_pawns),
    )


def serve_gate_reason(gate: dict[str, Any]) -> str:
    """Short human reason for logs / PromotionDecision."""
    if gate.get("missing_disagree"):
        return "missing disagree metrics; not serving child."
    if gate.get("passes_serve_gate"):
        parts = [
            f"disagree +{float(gate['d_disagree']):.4f}",
            f"overall {float(gate['d_top1']):+.4f}",
            f"pawn_regress {float(gate['pawn_gap_regress']):.4f}",
        ]
        if gate.get("ll_warn"):
            parts.append("ll_warn")
        return "serve gate pass: " + ", ".join(parts)
    fails: list[str] = []
    if not gate.get("pass_disagree"):
        fails.append(
            f"disagree lift {float(gate['d_disagree']):+.4f} < +{SERVE_GATE_MIN_DISAGREE_LIFT:.2f}"
        )
    if not gate.get("pass_overall"):
        fails.append(
            f"overall drop {float(gate['d_top1']):+.4f} "
            f"worse than -{SERVE_GATE_MAX_OVERALL_DROP:.2f}"
        )
    if not gate.get("pass_pawn"):
        fails.append(
            f"pawn regress {float(gate['pawn_gap_regress']):.4f} "
            f"> {SERVE_GATE_MAX_PAWN_GAP_REGRESS:.2f}"
        )
    return "serve gate fail: " + "; ".join(fails)
