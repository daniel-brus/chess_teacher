"""Create and inspect per-user pipeline Jobs in Kubernetes."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from chess_teacher.utils.env_utils import get_env_variable, get_optional_env_variable
from chess_teacher.utils.general_utils import generate_hash
from chess_teacher.utils.logging import get_logger

logger = get_logger()

APP_LABEL_KEY = "app"
APP_LABEL_VALUE = "chess-teacher"
LABEL_JOB_TYPE = "chess-teacher.io/job-type"
LABEL_WORK_ITEM = "chess-teacher.io/work-item"
JOB_TYPE_PIPELINE = "pipeline"

_PIPELINE_JOB_TEMPLATE = Path(__file__).resolve().parent / "k8s" / "job" / "pipeline.yaml"
_NAME_RE = re.compile(r"[^a-z0-9-]+")
_MAX_NAME_LEN = 63


def work_item_key(user_id: str) -> str:
    return generate_hash([user_id])[:32]


def pipeline_job_name(user_id: str, *, now: datetime | None = None) -> str:
    """Build a unique, DNS-safe Job name for one user's pipeline run."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d%H%M%S")
    base = _sanitize_job_name(f"pipeline-{user_id[:12]}-{stamp}")
    return base[:_MAX_NAME_LEN].rstrip("-")


def _sanitize_job_name(value: str) -> str:
    cleaned = _NAME_RE.sub("-", value.lower()).strip("-")
    return cleaned or "job"


@lru_cache(maxsize=1)
def kubernetes_client() -> tuple[Any, Any]:
    """Load the in-cluster or current kubeconfig client."""
    from kubernetes import client, config  # type: ignore[import-untyped]
    from kubernetes.client.rest import ApiException  # type: ignore[import-untyped]

    if get_optional_env_variable("KUBERNETES_SERVICE_HOST"):
        config.load_incluster_config()
    else:
        config.load_kube_config()
    return client, ApiException


def render_pipeline_job_manifest(
    *,
    job_name: str,
    user_id: str,
    work_item: str,
    image: str,
    image_pull_policy: str,
) -> dict[str, Any]:
    """Render the shared pipeline Job template with runtime values."""
    if not _PIPELINE_JOB_TEMPLATE.is_file():
        raise FileNotFoundError(f"Pipeline job template not found: {_PIPELINE_JOB_TEMPLATE}")
    rendered = (
        _PIPELINE_JOB_TEMPLATE
        .read_text(encoding="utf-8")
        .replace("REPLACE_JOB_NAME", job_name)
        .replace("REPLACE_USER_ID", user_id)
        .replace("REPLACE_WORK_ITEM", work_item)
        .replace("REPLACE_PIPELINE_JOB_IMAGE", image)
        .replace("REPLACE_IMAGE_PULL_POLICY", image_pull_policy)
    )
    manifest = yaml.safe_load(rendered)
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid pipeline job template: {_PIPELINE_JOB_TEMPLATE}")
    return manifest


def list_active_pipeline_jobs(*, namespace: str) -> set[str]:
    """Return work-item keys that already have an active pipeline Job."""
    client, api_exception = kubernetes_client()
    try:
        jobs = _list_pipeline_jobs(client, namespace)
    except api_exception as exc:
        logger.log_and_raise(exc, f"Failed to list pipeline jobs in namespace {namespace!r}")
    active: set[str] = set()

    for job in jobs:
        if not job.status or not job.status.active:
            continue
        labels = job.metadata.labels if job.metadata and job.metadata.labels else {}
        work_item = labels.get(LABEL_WORK_ITEM)
        if work_item:
            active.add(work_item)
    return active


def list_user_pipeline_jobs(*, namespace: str, user_id: str) -> list[Any]:
    """Return this user's pipeline Jobs, newest first."""
    client, api_exception = kubernetes_client()
    selector = ",".join((
        f"{APP_LABEL_KEY}={APP_LABEL_VALUE}",
        f"{LABEL_JOB_TYPE}={JOB_TYPE_PIPELINE}",
        f"{LABEL_WORK_ITEM}={work_item_key(user_id)}",
    ))
    try:
        jobs = client.BatchV1Api().list_namespaced_job(
            namespace=namespace,
            label_selector=selector,
        )
    except api_exception as exc:
        logger.log_and_raise(exc, f"Failed to list pipeline jobs for user {user_id!r}")
    return sorted(
        jobs.items,
        key=lambda job: job.metadata.creation_timestamp or datetime.min.replace(tzinfo=UTC),
        reverse=True,
    )


def get_pipeline_job(*, namespace: str, job_name: str) -> Any:
    """Return one pipeline Job by name."""
    client, api_exception = kubernetes_client()
    try:
        return client.BatchV1Api().read_namespaced_job(name=job_name, namespace=namespace)
    except api_exception as exc:
        logger.log_and_raise(exc, f"Failed to read pipeline job {job_name!r}")


def create_pipeline_job(
    *,
    namespace: str,
    user_id: str,
    image: str | None = None,
    image_pull_policy: str | None = None,
) -> str:
    """Create a Kubernetes Job for a UI-triggered pipeline run."""
    client, api_exception = kubernetes_client()
    job_name = pipeline_job_name(user_id)
    manifest = render_pipeline_job_manifest(
        job_name=job_name,
        user_id=user_id,
        work_item=work_item_key(user_id),
        image=image or get_env_variable("PIPELINE_JOB_IMAGE"),
        image_pull_policy=image_pull_policy
        or get_optional_env_variable("IMAGE_PULL_POLICY", "Always"),
    )
    try:
        client.BatchV1Api().create_namespaced_job(namespace=namespace, body=manifest)
    except api_exception as exc:
        if exc.status == 409:
            logger.warning("Pipeline job %s already exists; returning existing job.", job_name)
            return job_name
        logger.log_and_raise(exc, f"Failed to create pipeline job {job_name!r}")
    logger.info("Created pipeline job %s for user=%s.", job_name, user_id)
    return job_name


def read_pipeline_job_logs(*, namespace: str, job_name: str) -> str:
    """Read the latest stdout/stderr output for a pipeline Job's pod."""
    client, api_exception = kubernetes_client()
    try:
        core_api = client.CoreV1Api()
        pods = core_api.list_namespaced_pod(
            namespace=namespace,
            label_selector=f"job-name={job_name}",
        )
    except api_exception as exc:
        logger.log_and_raise(exc, f"Failed to find the pod for pipeline job {job_name!r}")
    if not pods.items:
        return ""

    pod = max(
        pods.items,
        key=lambda item: item.metadata.creation_timestamp or datetime.min.replace(tzinfo=UTC),
    )
    container_statuses = pod.status.container_statuses if pod.status else None
    state = container_statuses[0].state if container_statuses else None
    if not state or not (state.running or state.terminated):
        return ""

    try:
        output = core_api.read_namespaced_pod_log(
            name=pod.metadata.name,
            namespace=namespace,
            container="pipeline",
            tail_lines=1000,
        )
    except api_exception as exc:
        logger.log_and_raise(exc, f"Failed to read logs for pipeline job {job_name!r}")
    return output or ""


def pipeline_job_state(job: Any) -> tuple[str, str | None]:
    """Return ``pending``, ``running``, ``succeeded``, or ``failed`` and its message."""
    status = job.status
    if status is None:
        return "pending", None
    for condition in status.conditions or ():
        if condition.status != "True":
            continue
        if condition.type == "Complete":
            return "succeeded", condition.message
        if condition.type == "Failed":
            return "failed", condition.message
    if status.active:
        return "running", None
    if status.succeeded:
        return "succeeded", None
    if status.failed:
        return "failed", None
    return "pending", None


def _list_pipeline_jobs(client: Any, namespace: str) -> list[Any]:
    jobs = client.BatchV1Api().list_namespaced_job(
        namespace=namespace,
        label_selector=f"{APP_LABEL_KEY}={APP_LABEL_VALUE},{LABEL_JOB_TYPE}={JOB_TYPE_PIPELINE}",
    )
    return jobs.items
