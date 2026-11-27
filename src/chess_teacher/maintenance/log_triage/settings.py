"""Read TRIAGE_* knobs from environment."""

from __future__ import annotations

import os
from dataclasses import dataclass

from chess_teacher.maintenance.log_triage.constants import (
    DEFAULT_CLOSED_LOOKBACK_DAYS,
    DEFAULT_DEBATE_BAND,
    DEFAULT_FLAP_DAYS,
    DEFAULT_FLAP_REOPENS,
    DEFAULT_GRACE_HOURS,
    DEFAULT_LOCK_TTL_MINUTES,
    DEFAULT_MAX_ACTS,
    DEFAULT_MAX_EVALS,
    DEFAULT_MAX_LOG_LINES,
    DEFAULT_OPEN_THRESHOLD,
)


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


@dataclass(frozen=True)
class TriageSettings:
    open_threshold: float = DEFAULT_OPEN_THRESHOLD
    debate_band: float = DEFAULT_DEBATE_BAND
    max_acts: int = DEFAULT_MAX_ACTS
    max_evals: int = DEFAULT_MAX_EVALS
    max_log_lines: int = DEFAULT_MAX_LOG_LINES
    grace_hours: int = DEFAULT_GRACE_HOURS
    closed_lookback_days: int = DEFAULT_CLOSED_LOOKBACK_DAYS
    flap_reopens: int = DEFAULT_FLAP_REOPENS
    flap_days: int = DEFAULT_FLAP_DAYS
    lock_ttl_minutes: int = DEFAULT_LOCK_TTL_MINUTES


def load_settings() -> TriageSettings:
    return TriageSettings(
        open_threshold=_float("TRIAGE_OPEN_THRESHOLD", DEFAULT_OPEN_THRESHOLD),
        debate_band=_float("TRIAGE_DEBATE_BAND", DEFAULT_DEBATE_BAND),
        max_acts=_int("TRIAGE_MAX_ACTS", DEFAULT_MAX_ACTS),
        max_evals=_int("TRIAGE_MAX_EVALS", DEFAULT_MAX_EVALS),
        max_log_lines=_int("TRIAGE_MAX_LOG_LINES", DEFAULT_MAX_LOG_LINES),
        grace_hours=_int("TRIAGE_GRACE_HOURS", DEFAULT_GRACE_HOURS),
        closed_lookback_days=_int("TRIAGE_CLOSED_LOOKBACK_DAYS", DEFAULT_CLOSED_LOOKBACK_DAYS),
        flap_reopens=_int("TRIAGE_FLAP_REOPENS", DEFAULT_FLAP_REOPENS),
        flap_days=_int("TRIAGE_FLAP_DAYS", DEFAULT_FLAP_DAYS),
        lock_ttl_minutes=_int("TRIAGE_LOCK_TTL_MINUTES", DEFAULT_LOCK_TTL_MINUTES),
    )
