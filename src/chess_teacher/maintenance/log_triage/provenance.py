"""Capture triage git identity and best-effort observed deploy refs."""

from __future__ import annotations

import subprocess
from typing import Any


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    text = (result.stdout or "").strip()
    return text or None


def collect_provenance(
    *,
    observed_ref: str | None = None,
    observed_image_tag: str | None = None,
    observed_deploy_at: str | None = None,
) -> dict[str, Any]:
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    sha = _git("rev-parse", "HEAD")
    return {
        "triage_git_branch": branch,
        "triage_git_sha": sha,
        "observed_ref": observed_ref or "main@unknown",
        "observed_image_tag": observed_image_tag,
        "observed_deploy_at": observed_deploy_at,
    }
