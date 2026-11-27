"""Short-TTL single-flight locks for triage sources."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from chess_teacher.maintenance.log_triage.constants import DEFAULT_LOCK_TTL_MINUTES, LOCK_PREFIX
from chess_teacher.utils.object_storage.base import ObjectStorage


def lock_key(source: str) -> str:
    return f"{LOCK_PREFIX}/{source}.lock.json"


def _now() -> datetime:
    return datetime.now(UTC)


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def acquire_lock(
    storage: ObjectStorage,
    source: str,
    *,
    holder: str,
    ttl_minutes: int = DEFAULT_LOCK_TTL_MINUTES,
) -> dict[str, Any]:
    """Try to acquire lock. Returns ``{acquired: bool, lock: ...}``."""
    key = lock_key(source)
    raw = storage.read_bytes(key)
    now = _now()
    if raw is not None:
        existing = json.loads(raw.decode("utf-8"))
        expires = _parse_ts(existing["expires_at"])
        if expires > now and existing.get("holder") != holder:
            return {"acquired": False, "lock": existing, "reason": "held_by_other"}
    expires_at = (now + timedelta(minutes=ttl_minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")
    lock = {
        "source": source,
        "holder": holder,
        "acquired_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "expires_at": expires_at,
    }
    storage.write_text_atomic(
        key,
        json.dumps(lock, indent=2) + "\n",
        overwrite=True,
    )
    return {"acquired": True, "lock": lock}


def release_lock(
    storage: ObjectStorage,
    source: str,
    *,
    holder: str,
) -> dict[str, Any]:
    key = lock_key(source)
    raw = storage.read_bytes(key)
    if raw is None:
        return {"released": True, "reason": "already_absent"}
    existing = json.loads(raw.decode("utf-8"))
    if existing.get("holder") != holder:
        return {"released": False, "reason": "holder_mismatch", "lock": existing}
    storage.delete(key)
    return {"released": True, "reason": "deleted"}
