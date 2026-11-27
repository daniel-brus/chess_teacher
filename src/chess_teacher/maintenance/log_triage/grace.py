"""Grace-window helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def is_grace_blocked(
    grace_entries: list[dict[str, Any]],
    *,
    signature: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = now or datetime.now(UTC)
    for entry in grace_entries:
        if entry.get("signature") != signature:
            continue
        until = _parse(str(entry["suppress_until"]))
        if until > current:
            return {
                "blocked": True,
                "issue": entry.get("issue"),
                "suppress_until": entry["suppress_until"],
                "reason": entry.get("reason"),
            }
    return {"blocked": False}


def make_grace_entry(
    *,
    issue: int,
    signature: str,
    grace_hours: int,
    reason: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = now or datetime.now(UTC)
    until = current + timedelta(hours=grace_hours)
    return {
        "issue": issue,
        "signature": signature,
        "suppress_until": until.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reason": reason,
    }
