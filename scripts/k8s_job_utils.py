"""Shared helpers for rendering and applying Kubernetes Job manifests."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

_NAME_RE = re.compile(r"[^a-z0-9-]+")
_MAX_NAME_LEN = 63

REPO_ROOT = Path(__file__).resolve().parent.parent


def sanitize_job_name(value: str) -> str:
    cleaned = _NAME_RE.sub("-", value.lower()).strip("-")
    return cleaned or "job"


def truncate_job_name(value: str) -> str:
    return sanitize_job_name(value)[:_MAX_NAME_LEN].rstrip("-")


def load_job_template(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Job template not found: {path}")
    return path.read_text(encoding="utf-8")


def render_script_args_yaml(script_args: list[str]) -> str:
    """Render container args as a YAML flow-style list for template substitution."""
    rendered = yaml.dump(script_args, default_flow_style=True).strip()
    return rendered or "[]"


def kubectl_apply_manifest(manifest: dict[str, Any], *, namespace: str | None = None) -> None:
    payload = yaml.safe_dump(manifest, sort_keys=False)
    cmd = ["kubectl", "apply", "-f", "-"]
    if namespace:
        cmd.extend(["-n", namespace])
    result = subprocess.run(
        cmd,
        input=payload,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "kubectl apply failed"
        raise RuntimeError(detail)


def resolve_pipeline_job_image(*, namespace: str, override: str | None = None) -> str:
    import os

    if override:
        return override
    from_env = os.getenv("PIPELINE_JOB_IMAGE", "").strip()
    if from_env:
        return from_env
    result = subprocess.run(
        [
            "kubectl",
            "get",
            "cm",
            "chess-teacher-config",
            "-n",
            namespace,
            "-o",
            "jsonpath={.data.PIPELINE_JOB_IMAGE}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    image = result.stdout.strip()
    if result.returncode == 0 and image:
        return image
    raise SystemExit(
        "PIPELINE_JOB_IMAGE is not set and could not be read from "
        f"configmap/chess-teacher-config in namespace {namespace!r}. "
        "Export PIPELINE_JOB_IMAGE or pass --image."
    )
