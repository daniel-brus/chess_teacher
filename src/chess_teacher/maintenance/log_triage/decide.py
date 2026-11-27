"""Pure gate: confidence + policy → act | hold."""

from __future__ import annotations

from dataclasses import dataclass

from chess_teacher.maintenance.log_triage.constants import ACT_ACTIONS


@dataclass(frozen=True)
class DecisionInput:
    action: str
    confidence: float
    open_threshold: float
    grace_blocked: bool = False
    flap_locked: bool = False
    acts_so_far: int = 0
    max_acts: int = 5
    evals_so_far: int = 0
    max_evals: int = 30


@dataclass(frozen=True)
class DecisionResult:
    decision: str  # act | hold
    reasons: list[str]


def decide(inp: DecisionInput) -> DecisionResult:
    reasons: list[str] = []
    if inp.evals_so_far >= inp.max_evals:
        reasons.append("max_evals_reached")
        return DecisionResult(decision="hold", reasons=reasons)

    if inp.action not in ACT_ACTIONS:
        reasons.append(f"action_not_actable:{inp.action}")
        return DecisionResult(decision="hold", reasons=reasons)

    if inp.grace_blocked:
        reasons.append("grace_blocked")
        return DecisionResult(decision="hold", reasons=reasons)

    if inp.flap_locked:
        reasons.append("flap_locked")
        return DecisionResult(decision="hold", reasons=reasons)

    if inp.confidence < inp.open_threshold:
        reasons.append(f"confidence_below_threshold:{inp.confidence:.3f}<{inp.open_threshold:.3f}")
        return DecisionResult(decision="hold", reasons=reasons)

    if inp.acts_so_far >= inp.max_acts:
        reasons.append("max_acts_reached")
        return DecisionResult(decision="hold", reasons=reasons)

    reasons.append("passed_all_gates")
    return DecisionResult(decision="act", reasons=reasons)
