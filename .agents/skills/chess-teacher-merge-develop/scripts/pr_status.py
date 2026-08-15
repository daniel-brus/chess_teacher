#!/usr/bin/env python3
"""Read-only PR status helper for feature-to-develop merges (JSON on stdout)."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from typing import Any

_BUMP_LABELS = frozenset({"bump:patch", "bump:minor", "bump:major"})


def _gh_json(args: list[str]) -> Any:
    gh = shutil.which("gh")
    if gh is None:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "gh CLI not found on PATH",
                    "gh_args": args,
                },
                indent=2,
            )
        )
        raise SystemExit(1)

    result = subprocess.run(
        [gh, *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        payload = {
            "ok": False,
            "error": (result.stderr or result.stdout or "gh failed").strip(),
            "gh_args": args,
        }
        print(json.dumps(payload, indent=2))
        raise SystemExit(1)
    text = result.stdout.strip()
    return json.loads(text) if text else None


def _status(*, pr: str | None) -> dict[str, Any]:
    view_args = [
        "pr",
        "view",
        *([pr] if pr else []),
        "--json",
        "number,url,title,state,baseRefName,headRefName,isDraft,mergeable,mergeStateStatus,labels",
    ]
    pr_data = _gh_json(view_args)
    assert isinstance(pr_data, dict)

    labels = [item["name"] for item in pr_data.get("labels") or []]
    bump = [name for name in labels if name in _BUMP_LABELS]

    checks_args = [
        "pr",
        "checks",
        str(pr_data["number"]),
        "--json",
        "name,state,bucket,workflow,link",
    ]
    checks_raw = _gh_json(checks_args)
    checks = checks_raw if isinstance(checks_raw, list) else []

    buckets: dict[str, int] = {}
    for check in checks:
        bucket = str(check.get("bucket") or check.get("state") or "unknown")
        buckets[bucket] = buckets.get(bucket, 0) + 1

    failing = [
        {
            "name": check.get("name"),
            "state": check.get("state"),
            "bucket": check.get("bucket"),
            "workflow": check.get("workflow"),
            "link": check.get("link"),
        }
        for check in checks
        if str(check.get("bucket") or "").lower() in {"fail", "failed"}
        or str(check.get("state") or "").upper() in {"FAILURE", "ERROR", "CANCELLED"}
    ]

    pending = [
        check.get("name")
        for check in checks
        if str(check.get("bucket") or "").lower() in {"pending", "running"}
        or str(check.get("state") or "").upper() in {"PENDING", "IN_PROGRESS", "QUEUED"}
    ]

    return {
        "ok": True,
        "number": pr_data.get("number"),
        "url": pr_data.get("url"),
        "title": pr_data.get("title"),
        "state": pr_data.get("state"),
        "base": pr_data.get("baseRefName"),
        "head": pr_data.get("headRefName"),
        "is_draft": pr_data.get("isDraft"),
        "mergeable": pr_data.get("mergeable"),
        "merge_state_status": pr_data.get("mergeStateStatus"),
        "labels": labels,
        "bump_labels": bump,
        "bump_ok": len(bump) == 1,
        "checks_bucket_counts": buckets,
        "failing_checks": failing,
        "pending_checks": pending,
        "ci_green": not failing and not pending and len(checks) > 0,
        "ready_to_merge": (
            pr_data.get("baseRefName") == "develop"
            and pr_data.get("state") == "OPEN"
            and not pr_data.get("isDraft")
            and len(bump) == 1
            and not failing
            and not pending
            and len(checks) > 0
            and str(pr_data.get("mergeable") or "").upper() in {"MERGEABLE", "UNKNOWN"}
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Required; emit JSON.")
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status", help="PR + checks rollup for current branch or --pr")
    status.add_argument("--pr", help="PR number or URL (default: current branch PR)")
    args = parser.parse_args()

    if not args.json:
        print("Pass --json", file=sys.stderr)
        return 2

    if args.command == "status":
        print(json.dumps(_status(pr=args.pr), indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
