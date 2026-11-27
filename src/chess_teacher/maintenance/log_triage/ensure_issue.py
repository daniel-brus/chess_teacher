"""Create / comment / reopen GitHub issues via ``gh`` (optional dry-run)."""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any


def _gh(args: list[str]) -> tuple[int, str, str]:
    gh = shutil.which("gh")
    if gh is None:
        return 127, "", "gh CLI not found on PATH"
    result = subprocess.run(
        [gh, *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout or "", result.stderr or ""


def find_issues_by_signature(
    signature: str,
    *,
    lookback_days: int = 60,
    include_closed: bool = True,
) -> dict[str, Any]:
    """Search issues whose body contains the fingerprint marker."""
    marker = f"agent-fingerprint: {signature}"
    # GitHub search is imperfect for body HTML comments; also search visible field.
    queries = [
        f'repo:{{REPO}} in:body "{signature}"',
    ]
    # Caller may not know repo — use gh default repo from cwd.
    search_q = f'in:body "{signature}"'
    if not include_closed:
        search_q += " is:open"
    else:
        search_q += f" created:>={_lookback_date(lookback_days)}"

    code, out, err = _gh([
        "issue",
        "list",
        "--search",
        search_q,
        "--json",
        "number,title,state,url,labels,body",
        "--limit",
        "20",
    ])
    if code != 0:
        return {"ok": False, "error": err.strip() or out.strip(), "issues": [], "marker": marker}
    try:
        issues = json.loads(out) if out.strip() else []
    except json.JSONDecodeError:
        return {"ok": False, "error": "invalid gh json", "issues": [], "raw": out}
    matched = []
    for issue in issues:
        body = issue.get("body") or ""
        if signature in body:
            matched.append(issue)
    return {"ok": True, "issues": matched, "marker": marker, "queries": queries}


def _lookback_date(days: int) -> str:
    from datetime import UTC, datetime, timedelta

    return (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%d")


def build_issue_body(
    *,
    signature: str,
    title_hint: str,
    rationale: str,
    eval_id: str,
    provenance: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> str:
    evidence_lines = []
    for item in evidence[:5]:
        evidence_lines.append(f"- `{item.get('ref')}` @ `{item.get('ts')}`: {item.get('snippet')}")
    evid = "\n".join(evidence_lines) if evidence_lines else "_none_"
    return "\n".join([
        f"<!-- agent-fingerprint: {signature} -->",
        f"**Fingerprint:** `{signature}`",
        f"**Eval:** `{eval_id}`",
        f"**Triage:** `{provenance.get('triage_git_branch')}@"
        f"{str(provenance.get('triage_git_sha') or '')[:7]}`",
        f"**Observed:** `{provenance.get('observed_ref')}`",
        "",
        "## Summary",
        title_hint,
        "",
        "## Rationale",
        rationale,
        "",
        "## Evidence",
        evid,
    ])


def ensure_issue(
    *,
    action: str,
    signature: str,
    title: str,
    body: str,
    labels: list[str],
    existing_issue: int | None = None,
    comment: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Create, comment, or reopen an issue. ``action``: create|comment|reopen."""
    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "action": action,
            "signature": signature,
            "title": title,
            "labels": labels,
            "existing_issue": existing_issue,
            "comment": comment,
            "body_preview": body[:500],
        }

    if action == "create":
        args = ["issue", "create", "--title", title, "--body", body]
        for label in labels:
            args.extend(["--label", label])
        code, out, err = _gh(args)
        if code != 0:
            return {"ok": False, "error": err.strip() or out.strip(), "action": action}
        return {"ok": True, "action": action, "url": out.strip()}

    if existing_issue is None:
        return {"ok": False, "error": "existing_issue required", "action": action}

    if action == "reopen":
        code, out, err = _gh(["issue", "reopen", str(existing_issue)])
        if code != 0:
            return {"ok": False, "error": err.strip() or out.strip(), "action": action}
        for label in labels:
            _gh(["issue", "edit", str(existing_issue), "--add-label", label])
        if comment:
            _gh(["issue", "comment", str(existing_issue), "--body", comment])
        return {"ok": True, "action": action, "issue": existing_issue, "stdout": out.strip()}

    if action == "comment":
        code, out, err = _gh(["issue", "comment", str(existing_issue), "--body", comment or body])
        if code != 0:
            return {"ok": False, "error": err.strip() or out.strip(), "action": action}
        return {"ok": True, "action": action, "issue": existing_issue, "stdout": out.strip()}

    return {"ok": False, "error": f"unknown action {action}", "action": action}
