"""Render and apply a one-off script Job in Kubernetes (prod ops).

On the production VPS prefer the CD-deployed wrapper (no repo / Python needed)::

    /opt/chess_teacher/k8s/run-script-job.sh baseline_training

From a repo checkout with kubectl (local k3d or laptop with kubeconfig)::

    python scripts/run_script_job.py baseline_training --dry-run
    python scripts/run_script_job.py baseline_promotion -- --wait

Image and pull policy are resolved automatically from the live streamlit
deployment / configmap unless you pass ``--image`` / ``--image-pull-policy``.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from scripts.k8s_job_utils import (
    REPO_ROOT,
    kubectl_apply_manifest,
    load_job_template,
    render_script_args_yaml,
    resolve_image_pull_policy,
    resolve_pipeline_job_image,
    truncate_job_name,
)

# Keep in sync with orchestration/k8s/run-script-job.sh ALLOWED_SCRIPTS.
ALLOWED_SCRIPTS = frozenset({
    "baseline_training.py",
    "baseline_promotion.py",
    "maintenance.py",
})

_SCRIPT_JOB_TEMPLATE = REPO_ROOT / "orchestration" / "k8s" / "job" / "script.yaml"
_DEFAULT_NAMESPACE = "chess-teacher"


def normalize_script_basename(name: str) -> str:
    base = Path(name.strip()).name
    if not base:
        raise SystemExit("Script name must not be empty.")
    if not base.endswith(".py"):
        base = f"{base}.py"
    return base


def validate_script_args(script_args: list[str]) -> list[str]:
    for arg in script_args:
        if not arg:
            raise SystemExit("Script arguments must not be empty strings.")
        # Block shell metacharacters even though kubectl applies YAML (defense in depth).
        if any(ch in arg for ch in ";|&`$<>(){}\\\r\n\x00"):
            raise SystemExit(f"Disallowed characters in script argument: {arg!r}")
    return script_args


def script_job_name(script_basename: str, *, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d%H%M%S")
    stem = Path(script_basename).stem
    return truncate_job_name(f"script-{stem}-{stamp}")


def render_script_job_manifest(
    *,
    job_name: str,
    script_basename: str,
    script_args: list[str],
    image: str,
    image_pull_policy: str,
    namespace: str,
) -> dict[str, Any]:
    rendered = (
        load_job_template(_SCRIPT_JOB_TEMPLATE)
        .replace("REPLACE_JOB_NAME", job_name)
        .replace("REPLACE_SCRIPT_JOB_IMAGE", image)
        .replace("REPLACE_IMAGE_PULL_POLICY", image_pull_policy)
        .replace("REPLACE_SCRIPT_BASENAME", script_basename)
        .replace("REPLACE_SCRIPT_ARGS", render_script_args_yaml(script_args))
    )
    manifest = yaml.safe_load(rendered)
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid script job template: {_SCRIPT_JOB_TEMPLATE}")
    if manifest.get("metadata", {}).get("namespace") != namespace:
        manifest.setdefault("metadata", {})["namespace"] = namespace
    return manifest


def wait_for_job(
    job_name: str,
    *,
    namespace: str,
    timeout_seconds: int = 86_400,
    poll_seconds: float = 10.0,
) -> str:
    import subprocess

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = subprocess.run(
            [
                "kubectl",
                "get",
                "job",
                job_name,
                "-n",
                namespace,
                "-o",
                "jsonpath={.status.succeeded} {.status.failed}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"Failed to get job {job_name!r}")
        parts = result.stdout.strip().split()
        succeeded = parts[0] if parts else ""
        failed = parts[1] if len(parts) > 1 else ""
        if succeeded and succeeded != "0":
            return "Complete"
        if failed and failed != "0":
            return "Failed"
        time.sleep(poll_seconds)
    raise TimeoutError(f"Timed out waiting for job {job_name!r} in namespace {namespace!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render and apply a whitelisted scripts/*.py Job in Kubernetes. "
            "On the VPS prefer /opt/chess_teacher/k8s/run-script-job.sh instead."
        ),
    )
    parser.add_argument(
        "script",
        help="Script entrypoint basename (e.g. baseline_training or baseline_training.py)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print rendered YAML without applying.",
    )
    parser.add_argument(
        "--namespace",
        default=_DEFAULT_NAMESPACE,
        help=f"Kubernetes namespace (default: {_DEFAULT_NAMESPACE}).",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Container image override (default: streamlit deploy / PIPELINE_JOB_IMAGE).",
    )
    parser.add_argument(
        "--image-pull-policy",
        default=None,
        help="Image pull policy override (default: configmap / IMAGE_PULL_POLICY / Always).",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Poll until the Job completes or fails.",
    )
    return parser


def split_script_args(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" in argv:
        idx = argv.index("--")
        return argv[:idx], argv[idx + 1 :]
    return argv, []


def main(argv: list[str] | None = None) -> int:
    cli_argv, script_args = split_script_args(list(argv if argv is not None else sys.argv[1:]))
    parser = build_parser()
    args = parser.parse_args(cli_argv)

    script_basename = normalize_script_basename(args.script)
    if script_basename not in ALLOWED_SCRIPTS:
        allowed = ", ".join(sorted(ALLOWED_SCRIPTS))
        raise SystemExit(f"Script {script_basename!r} is not whitelisted. Allowed: {allowed}")

    validate_script_args(script_args)

    job_name = script_job_name(script_basename)
    image = resolve_pipeline_job_image(namespace=args.namespace, override=args.image)
    image_pull_policy = resolve_image_pull_policy(
        namespace=args.namespace,
        override=args.image_pull_policy,
    )

    manifest = render_script_job_manifest(
        job_name=job_name,
        script_basename=script_basename,
        script_args=script_args,
        image=image,
        image_pull_policy=image_pull_policy,
        namespace=args.namespace,
    )

    if args.dry_run:
        print(yaml.safe_dump(manifest, sort_keys=False), end="")
        return 0

    kubectl_apply_manifest(manifest, namespace=args.namespace)

    print(f"Created Job {job_name} in namespace {args.namespace}.")
    print(f"  image: {image}")
    print(f"  kubectl logs -n {args.namespace} job/{job_name} -f")
    print(f"  kubectl delete job {job_name} -n {args.namespace}")

    if args.wait:
        status = wait_for_job(job_name, namespace=args.namespace)
        print(f"Job {job_name} finished with status: {status}")
        return 0 if status == "Complete" else 1

    return 0


if __name__ == "__main__":
    from chess_teacher.utils.process_utils import run_script_main

    run_script_main(main)
