"""Deterministic helpers for agent log triage (state, signatures, decide, audit)."""

from __future__ import annotations

from chess_teacher.maintenance.log_triage.constants import (
    AUDIT_PREFIX,
    GLOBAL_KNOWN_KEY,
    LOCK_PREFIX,
    STATE_PREFIX,
    TRIAGE_PREFIX,
)
from chess_teacher.maintenance.log_triage.decide import DecisionInput, decide
from chess_teacher.maintenance.log_triage.eval_ids import make_eval_id
from chess_teacher.maintenance.log_triage.redact import redact_snippet
from chess_teacher.maintenance.log_triage.signature import coarse_signature_from_log

__all__ = [
    "AUDIT_PREFIX",
    "GLOBAL_KNOWN_KEY",
    "LOCK_PREFIX",
    "STATE_PREFIX",
    "TRIAGE_PREFIX",
    "DecisionInput",
    "coarse_signature_from_log",
    "decide",
    "make_eval_id",
    "redact_snippet",
]
