"""Write evaluation audit objects."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from chess_teacher.maintenance.log_triage.constants import AUDIT_PREFIX
from chess_teacher.maintenance.log_triage.redact import redact_snippet
from chess_teacher.utils.object_storage.base import ObjectStorage


def audit_key(eval_id: str, *, when: datetime | None = None) -> str:
    ts = when or datetime.now(UTC)
    return f"{AUDIT_PREFIX}/{ts:%Y}/{ts:%m}/{ts:%d}/{eval_id}.json"


def write_eval_audit(
    storage: ObjectStorage,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Redact evidence snippets and persist audit JSON. Returns key + payload."""
    payload = dict(evaluation)
    evidence = []
    for item in payload.get("evidence") or []:
        if not isinstance(item, dict):
            continue
        cloned = dict(item)
        if "snippet" in cloned and isinstance(cloned["snippet"], str):
            cloned["snippet"] = redact_snippet(cloned["snippet"])
        evidence.append(cloned)
    payload["evidence"] = evidence
    if isinstance(payload.get("rationale"), str):
        payload["rationale"] = redact_snippet(payload["rationale"], max_len=2000)
    debate = payload.get("debate")
    if isinstance(debate, dict):
        for side in ("advocate_high", "advocate_low"):
            block = debate.get(side)
            if isinstance(block, dict) and isinstance(block.get("argument"), str):
                block["argument"] = redact_snippet(block["argument"], max_len=2000)
        if isinstance(debate.get("judge_rationale"), str):
            debate["judge_rationale"] = redact_snippet(debate["judge_rationale"], max_len=2000)

    eval_id = str(payload["eval_id"])
    key = audit_key(eval_id)
    storage.write_text_atomic(
        key,
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        overwrite=True,
    )
    return {"key": key, "evaluation": payload}
