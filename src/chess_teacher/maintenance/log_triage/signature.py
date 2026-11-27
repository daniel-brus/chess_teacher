"""Coarse signatures for log / GHA candidates."""

from __future__ import annotations

import re
from typing import Any

_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_HEX_RE = re.compile(r"\b[0-9a-f]{16,}\b", re.IGNORECASE)
_NUM_RE = re.compile(r"\b\d{4,}\b")
_POD_RE = re.compile(
    r"\b([a-z0-9-]+)-[0-9a-f]{5,10}-[a-z0-9]{5}\b",
    re.IGNORECASE,
)


def _normalize_token(value: str | None) -> str:
    if not value:
        return "-"
    text = value.strip()
    text = _UUID_RE.sub("<uuid>", text)
    text = _POD_RE.sub(r"\1-<pod>", text)
    text = _HEX_RE.sub("<hex>", text)
    text = _NUM_RE.sub("<num>", text)
    text = re.sub(r"\s+", " ", text)
    return text[:120] if len(text) > 120 else text


def _top_app_module(traceback: str | None) -> str:
    if not traceback:
        return "-"
    for line in traceback.splitlines():
        if "src/chess_teacher/" in line.replace("\\", "/"):
            path = line.replace("\\", "/")
            idx = path.find("src/chess_teacher/")
            fragment = path[idx:].split(",")[0].strip()
            # File ".../src/chess_teacher/foo/bar.py", line N, in fn
            match = re.search(r"(src/chess_teacher/[^\"]+\.py)", fragment)
            if match:
                rel = match.group(1)
                return rel.removeprefix("src/").removesuffix(".py").replace("/", ".")
    return "-"


def coarse_signature_from_log(record: dict[str, Any], *, source: str = "s3") -> str:
    """Build ``v1|{source}|{logger}|{exc_type}|{module}`` from a JSON log record."""
    logger = _normalize_token(str(record.get("logger") or "-"))
    exc_type = _normalize_token(str(record.get("exc_type") or "ERROR"))
    module = _top_app_module(
        record.get("traceback") if isinstance(record.get("traceback"), str) else None
    )
    if module == "-" and logger != "-":
        module = logger.split(".")[-1] if "." in logger else logger
    return f"v1|{source}|{logger}|{exc_type}|{module}"


def coarse_signature_from_gha(
    *,
    workflow: str,
    job: str,
    step: str | None = None,
) -> str:
    """Build ``v1|gha|{workflow}|{job}|{step}``."""
    return (
        f"v1|gha|"
        f"{_normalize_token(workflow)}|"
        f"{_normalize_token(job)}|"
        f"{_normalize_token(step or '-')}"
    )
