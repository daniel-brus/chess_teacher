"""Paths and defaults for agent triage durable state."""

from __future__ import annotations

TRIAGE_PREFIX = "agent/triage"
STATE_PREFIX = f"{TRIAGE_PREFIX}/state"
LOCK_PREFIX = f"{TRIAGE_PREFIX}/locks"
AUDIT_PREFIX = f"{TRIAGE_PREFIX}/audit"
CONFIG_KEY = f"{TRIAGE_PREFIX}/config/settings.json"
GLOBAL_KNOWN_KEY = f"{STATE_PREFIX}/known_issues_global.json"

SOURCES = frozenset({"s3", "vps", "gha"})
KNOWN_ISSUE_STATUSES = frozenset({
    "open",
    "in_progress",
    "pr_open",
    "grace",
    "flap_locked",
    "watching",
})
EVAL_ACTIONS = frozenset({"open", "reopen", "comment_only", "wait", "skip_duplicate", "skip_noise"})
ACT_ACTIONS = frozenset({"open", "reopen"})

DEFAULT_OPEN_THRESHOLD = 0.70
DEFAULT_MAX_ACTS = 5
DEFAULT_MAX_EVALS = 30
DEFAULT_MAX_LOG_LINES = 500
DEFAULT_GRACE_HOURS = 6
DEFAULT_CLOSED_LOOKBACK_DAYS = 60
DEFAULT_FLAP_REOPENS = 3
DEFAULT_FLAP_DAYS = 14
DEFAULT_LOCK_TTL_MINUTES = 45
DEFAULT_DEBATE_BAND = 0.15

SCHEMA_VERSION = 1
