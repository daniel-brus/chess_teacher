"""Redact sensitive-looking tokens from evidence snippets."""

from __future__ import annotations

import re

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?i)(bearer\s+)[a-z0-9._\-]+"), r"\1<redacted>"),
    (re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)\S+"), r"\1<redacted>"),
    (re.compile(r"(?i)(password\s*[:=]\s*)\S+"), r"\1<redacted>"),
    (re.compile(r"(?i)(secret\s*[:=]\s*)\S+"), r"\1<redacted>"),
    (re.compile(r"(?i)(token\s*[:=]\s*)\S+"), r"\1<redacted>"),
    (
        re.compile(
            r"(?i)(-----BEGIN [A-Z ]*PRIVATE KEY-----)(.*?)(-----END [A-Z ]*PRIVATE KEY-----)",
            re.DOTALL,
        ),
        r"\1<redacted>\3",
    ),
    (
        re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
        "<email>",
    ),
]


def redact_snippet(text: str, *, max_len: int = 400) -> str:
    """Return a truncated, redacted snippet safe for Slack/issue/audit."""
    out = text or ""
    for pattern, repl in _PATTERNS:
        out = pattern.sub(repl, out)
    out = out.replace("\x00", "")
    if len(out) > max_len:
        out = out[: max_len - 3] + "..."
    return out
