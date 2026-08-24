"""Shared helpers for rendering and applying Kubernetes Job manifests."""

from __future__ import annotations

import os
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


def _kubectl_jsonpath(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["kubectl", *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def resolve_pipeline_job_image(*, namespace: str, override: str | None = None) -> str:
    """Resolve Job image: --image → env → streamlit deploy → configmap."""
    if override:
        return override
    from_env = os.getenv("PIPELINE_JOB_IMAGE", "").strip()
    if from_env:
        return from_env

    from_deploy = _kubectl_jsonpath([
        "get",
        "deploy",
        "streamlit",
        "-n",
        namespace,
        "-o",
        "jsonpath={.spec.template.spec.containers[0].image}",
    ])
    if from_deploy:
        return from_deploy

    from_cm = _kubectl_jsonpath([
        "get",
        "cm",
        "chess-teacher-config",
        "-n",
        namespace,
        "-o",
        "jsonpath={.data.PIPELINE_JOB_IMAGE}",
    ])
    if from_cm:
        return from_cm

    raise SystemExit(
        "Could not resolve container image. Pass --image, export PIPELINE_JOB_IMAGE, "
        f"or ensure deploy/streamlit (or configmap/chess-teacher-config) exists in "
        f"namespace {namespace!r}."
    )


def resolve_image_pull_policy(*, namespace: str, override: str | None = None) -> str:
    """Resolve pull policy: --image-pull-policy → env → configmap → Always."""
    if override:
        return override
    from_env = os.getenv("IMAGE_PULL_POLICY", "").strip()
    if from_env:
        return from_env
    from_cm = _kubectl_jsonpath([
        "get",
        "cm",
        "chess-teacher-config",
        "-n",
        namespace,
        "-o",
        "jsonpath={.data.IMAGE_PULL_POLICY}",
    ])
    return from_cm or "Always"
