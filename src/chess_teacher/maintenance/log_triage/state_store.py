"""Load/save triage state objects in object storage."""

from __future__ import annotations

import json
from typing import Any

from chess_teacher.maintenance.log_triage.constants import GLOBAL_KNOWN_KEY, STATE_PREFIX
from chess_teacher.maintenance.log_triage.schemas import (
    StateValidationError,
    empty_global_known,
    empty_source_state,
    validate_global_known,
    validate_source_state,
)
from chess_teacher.utils.object_storage.base import ObjectStorage


def state_key(source: str) -> str:
    return f"{STATE_PREFIX}/{source}.json"


def _read_json(storage: ObjectStorage, key: str) -> dict[str, Any] | None:
    raw = storage.read_bytes(key)
    if raw is None:
        return None
    return json.loads(raw.decode("utf-8"))


def load_source_state(
    storage: ObjectStorage,
    source: str,
    *,
    create_if_missing: bool = True,
) -> dict[str, Any]:
    key = state_key(source)
    doc = _read_json(storage, key)
    if doc is None:
        if not create_if_missing:
            raise FileNotFoundError(key)
        doc = empty_source_state(source=source)
        return validate_source_state(doc)
    return validate_source_state(doc)


def save_source_state(
    storage: ObjectStorage,
    doc: dict[str, Any],
    *,
    expected_updated_at: str | None,
) -> dict[str, Any]:
    """Write state with optimistic concurrency on ``updated_at``."""
    validated = validate_source_state(doc)
    source = validated["source"]
    key = state_key(source)
    existing = _read_json(storage, key)
    if existing is not None:
        current = existing.get("updated_at")
        if expected_updated_at is not None and current != expected_updated_at:
            raise StateValidationError(
                f"concurrency conflict for {source}: "
                f"expected updated_at={expected_updated_at!r} got {current!r}"
            )
    storage.write_text_atomic(
        key,
        json.dumps(validated, indent=2, ensure_ascii=False) + "\n",
        overwrite=True,
    )
    return validated


def load_global_known(
    storage: ObjectStorage,
    *,
    create_if_missing: bool = True,
) -> dict[str, Any]:
    doc = _read_json(storage, GLOBAL_KNOWN_KEY)
    if doc is None:
        if not create_if_missing:
            raise FileNotFoundError(GLOBAL_KNOWN_KEY)
        return empty_global_known()
    return validate_global_known(doc)


def save_global_known(storage: ObjectStorage, doc: dict[str, Any]) -> dict[str, Any]:
    validated = validate_global_known(doc)
    storage.write_text_atomic(
        GLOBAL_KNOWN_KEY,
        json.dumps(validated, indent=2, ensure_ascii=False) + "\n",
        overwrite=True,
    )
    return validated


def upsert_global_issue(
    doc: dict[str, Any],
    *,
    issue: int,
    signature: str,
    status: str,
    source: str,
    updated_at: str,
) -> dict[str, Any]:
    issues = list(doc.get("issues") or [])
    found = False
    for item in issues:
        if item.get("signature") == signature or item.get("issue") == issue:
            item["issue"] = issue
            item["signature"] = signature
            item["status"] = status
            item["updated_at"] = updated_at
            sources = list(item.get("sources") or [])
            if source not in sources:
                sources.append(source)
            item["sources"] = sources
            found = True
            break
    if not found:
        issues.append({
            "issue": issue,
            "signature": signature,
            "status": status,
            "sources": [source],
            "updated_at": updated_at,
        })
    doc["issues"] = issues
    return doc


def find_global_by_signature(doc: dict[str, Any], signature: str) -> dict[str, Any] | None:
    for item in doc.get("issues") or []:
        if item.get("signature") == signature:
            return item
    return None
