"""Stable evaluation ids for idempotent retries."""

from __future__ import annotations

import hashlib


def make_eval_id(
    *,
    run_id: str,
    signature: str,
    action: str,
    scan_from: str,
    scan_to: str,
) -> str:
    payload = f"{run_id}|{signature}|{action}|{scan_from}|{scan_to}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
