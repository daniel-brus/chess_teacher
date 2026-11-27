"""Validate / build triage state and related JSON documents."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from chess_teacher.maintenance.log_triage.constants import (
    KNOWN_ISSUE_STATUSES,
    SCHEMA_VERSION,
    SOURCES,
)


class StateValidationError(ValueError):
    """Raised when triage state JSON is invalid."""


def _require(obj: dict[str, Any], key: str, typ: type | tuple[type, ...]) -> Any:
    if key not in obj:
        raise StateValidationError(f"missing field: {key}")
    val = obj[key]
    if not isinstance(val, typ):
        raise StateValidationError(f"field {key} has wrong type: {type(val).__name__}")
    return val


def validate_source_state(doc: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(doc, dict):
        raise StateValidationError("state must be an object")
    version = doc.get("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise StateValidationError(f"unsupported schema_version: {version}")
    source = _require(doc, "source", str)
    if source not in SOURCES:
        raise StateValidationError(f"invalid source: {source}")
    _require(doc, "watermark_ts", str)
    _require(doc, "updated_at", str)
    _require(doc, "source_ok", bool)
    handoff = _require(doc, "handoff", dict)
    for key in ("narrative", "scan_from", "scan_to", "start_here"):
        _require(handoff, key, str)
    for key in ("watch", "uncertain"):
        items = _require(handoff, key, list)
        if not all(isinstance(x, str) for x in items):
            raise StateValidationError(f"handoff.{key} must be string list")
    known = _require(doc, "known_issues", list)
    for item in known:
        if not isinstance(item, dict):
            raise StateValidationError("known_issues entries must be objects")
        _require(item, "issue", int)
        _require(item, "signature", str)
        status = _require(item, "status", str)
        if status not in KNOWN_ISSUE_STATUSES:
            raise StateValidationError(f"invalid known_issues status: {status}")
        _require(item, "note", str)
    grace = _require(doc, "grace", list)
    for item in grace:
        if not isinstance(item, dict):
            raise StateValidationError("grace entries must be objects")
        _require(item, "issue", int)
        _require(item, "signature", str)
        _require(item, "suppress_until", str)
        _require(item, "reason", str)
    return doc


def empty_source_state(*, source: str, watermark_ts: str | None = None) -> dict[str, Any]:
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    wm = watermark_ts or "1970-01-01T00:00:00.000Z"
    return {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "watermark_ts": wm,
        "updated_at": now,
        "run_id": None,
        "provenance": {
            "triage_git_branch": None,
            "triage_git_sha": None,
            "observed_ref": "main@unknown",
            "observed_image_tag": None,
            "observed_deploy_at": None,
        },
        "source_ok": True,
        "source_error": None,
        "handoff": {
            "narrative": "Initial empty state. No prior triage run.",
            "scan_from": wm,
            "scan_to": wm,
            "start_here": "Scan ERROR logs since watermark and write the first real handoff.",
            "watch": [],
            "uncertain": [],
        },
        "known_issues": [],
        "grace": [],
    }


def empty_global_known() -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "issues": []}


def validate_global_known(doc: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(doc, dict):
        raise StateValidationError("global known_issues must be an object")
    issues = _require(doc, "issues", list)
    for item in issues:
        if not isinstance(item, dict):
            raise StateValidationError("global issues entries must be objects")
        _require(item, "issue", int)
        _require(item, "signature", str)
        _require(item, "status", str)
        sources = _require(item, "sources", list)
        if not all(isinstance(s, str) for s in sources):
            raise StateValidationError("sources must be string list")
        _require(item, "updated_at", str)
    return doc
