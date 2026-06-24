from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from chess_teacher.platform.user import User
from chess_teacher.utils.db.client import DatabaseClient, get_db_client
from chess_teacher.utils.env_utils import get_env_variable
from chess_teacher.utils.general_utils import generate_hash
from chess_teacher.utils.logging import get_logger

logger = get_logger()

APP_LABEL_KEY = "app"
APP_LABEL_VALUE = "chess-teacher"
LABEL_JOB_TYPE = "chess-teacher.io/job-type"
LABEL_WORK_ITEM = "chess-teacher.io/work-item"
JOB_TYPE_PIPELINE = "pipeline"

_PIPELINE_JOB_TEMPLATE = (
    Path(__file__).resolve().parent.parent / "orchestration" / "k8s" / "job" / "pipeline.yaml"
)

_NAME_RE = re.compile(r"[^a-z0-9-]+")
_MAX_NAME_LEN = 63


@dataclass(frozen=True)
class DispatchResult:
    scanned_users: int
    spawned_jobs: tuple[str, ...]
    skipped_active_job: int
    skipped_not_due: int
    skipped_cooldown: int


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


def _kubernetes_client():
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException

    if os.getenv("KUBERNETES_SERVICE_HOST"):
        config.load_incluster_config()
    else:
        config.load_kube_config()
    return client, ApiException


def _load_pipeline_job_template() -> str:
    if not _PIPELINE_JOB_TEMPLATE.is_file():
        logger.log_and_raise(
            FileNotFoundError(f"Pipeline job template not found: {_PIPELINE_JOB_TEMPLATE}")
        )
    return _PIPELINE_JOB_TEMPLATE.read_text(encoding="utf-8")


def render_pipeline_job_manifest(
    *,
    job_name: str,
    user_id: str,
    work_item: str,
    image: str,
    image_pull_policy: str,
) -> dict[str, Any]:
    """Render orchestration/k8s/job/pipeline.yaml with runtime values."""
    rendered = (
        _load_pipeline_job_template()
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
    client, api_exception = _kubernetes_client()
    batch_api = client.BatchV1Api()
    active: set[str] = set()

    try:
        jobs = batch_api.list_namespaced_job(
            namespace=namespace,
            label_selector=f"{APP_LABEL_KEY}={APP_LABEL_VALUE},{LABEL_JOB_TYPE}={JOB_TYPE_PIPELINE}",
        )
    except api_exception as exc:
        logger.log_and_raise(exc, f"Failed to list pipeline jobs in namespace {namespace!r}")

    for job in jobs.items:
        if not job.status or not job.status.active:
            continue
        labels = job.metadata.labels if job.metadata and job.metadata.labels else {}
        work_item = labels.get(LABEL_WORK_ITEM)
        if work_item:
            active.add(work_item)
    return active


def create_pipeline_job(
    *,
    namespace: str,
    user_id: str,
    image: str | None = None,
) -> str:
    """Create a Kubernetes Job from orchestration/k8s/job/pipeline.yaml for one user."""
    client, api_exception = _kubernetes_client()
    batch_api = client.BatchV1Api()

    job_image = image or get_env_variable("PIPELINE_JOB_IMAGE")
    image_pull_policy = os.getenv("IMAGE_PULL_POLICY", "Always")
    job_name = pipeline_job_name(user_id)
    item_key = work_item_key(user_id)

    job_body = render_pipeline_job_manifest(
        job_name=job_name,
        user_id=user_id,
        work_item=item_key,
        image=job_image,
        image_pull_policy=image_pull_policy,
    )

    try:
        batch_api.create_namespaced_job(namespace=namespace, body=job_body)
    except api_exception as exc:
        if exc.status == 409:
            logger.warning("Pipeline job %s already exists; skipping create.", job_name)
            return job_name
        logger.log_and_raise(exc, f"Failed to create pipeline job {job_name!r}")

    logger.info("Created pipeline job %s for user=%s.", job_name, user_id)
    return job_name


def dispatch_pipeline_jobs(
    *,
    db_client: DatabaseClient | None = None,
    namespace: str | None = None,
) -> DispatchResult:
    """
    Spawn parallel pipeline Jobs in Kubernetes for every user that does not
    already have an active Job.

    Intended to run inside the ingestion-dispatcher CronJob pod.
    """
    db = db_client or get_db_client()
    k8s_namespace = namespace or get_env_variable("K8S_NAMESPACE")
    now = datetime.now(UTC)

    users = User.fetch_all_from_db(db)
    active_jobs = list_active_pipeline_jobs(namespace=k8s_namespace)

    spawned: list[str] = []
    skipped_active = 0
    skipped_not_due = 0
    skipped_cooldown = 0

    for user in users:
        if not user.is_cron_due(now):
            skipped_not_due += 1
            continue
        if not user.pipeline_allowed_to_run(db):
            skipped_cooldown += 1
            logger.info(
                "Skipping user=%s: pipeline cooldown active.",
                user.user_id,
            )
            continue

        user_key = work_item_key(user.user_id)
        if user_key in active_jobs:
            skipped_active += 1
            logger.info(
                "Skipping user=%s: pipeline job already active.",
                user.user_id,
            )
            continue

        job_name = create_pipeline_job(
            namespace=k8s_namespace,
            user_id=user.user_id,
        )
        spawned.append(job_name)
        active_jobs.add(user_key)

    result = DispatchResult(
        scanned_users=len(users),
        spawned_jobs=tuple(spawned),
        skipped_active_job=skipped_active,
        skipped_not_due=skipped_not_due,
        skipped_cooldown=skipped_cooldown,
    )
    logger.info(
        "Dispatch finished: users=%s spawned=%s skipped_active=%s skipped_not_due=%s skipped_cooldown=%s",
        result.scanned_users,
        len(result.spawned_jobs),
        result.skipped_active_job,
        result.skipped_not_due,
        result.skipped_cooldown,
    )
    return result


def main() -> None:
    logger.info("Pipeline dispatcher started.")
    result = dispatch_pipeline_jobs()
    logger.info(
        "Pipeline dispatcher completed: spawned=%s jobs.",
        len(result.spawned_jobs),
    )


if __name__ == "__main__":
    main()
