from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from chess_teacher.platform.dispatch import (
    DEFER_ACTIVE_JOB,
    DEFER_MEMORY,
    DEFER_WAITING_TURN,
    pipeline_host_has_memory,
    plan_pipeline_dispatch,
)
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
    Path(__file__).resolve().parents[2] / "orchestration" / "k8s" / "job" / "pipeline.yaml"
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
    deferred: int


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
    from kubernetes import client, config  # type: ignore[import-untyped]
    from kubernetes.client.rest import ApiException  # type: ignore[import-untyped]

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
    """Start at most one user pipeline job, and queue the rest for the next tick.

    A user is eligible in their cron window, or later the same local day when a
    previous tick left ``pipeline_retry_at`` set. The next dispatcher run (30
    minutes later) tries that user again. Nobody new starts while a pipeline
    job is still running, or while MemAvailable is under the training minimum.

    Intended to run inside the ingestion-dispatcher CronJob pod.
    """
    db = db_client or get_db_client()
    k8s_namespace = namespace or get_env_variable("K8S_NAMESPACE")
    now = datetime.now(UTC)

    users = User.fetch_all_from_db(db)
    active_jobs = list_active_pipeline_jobs(namespace=k8s_namespace)
    users_by_id = {user.user_id: user for user in users}
    active_user_ids = {user.user_id for user in users if work_item_key(user.user_id) in active_jobs}
    cooldown_blocked = {
        user.user_id
        for user in users
        if user.is_pipeline_dispatch_due(now) and not user.pipeline_allowed_to_run(db)
    }
    due_count = sum(1 for user in users if user.is_pipeline_dispatch_due(now))
    memory_available = pipeline_host_has_memory()
    logger.info(
        "Dispatch scan: namespace=%s users=%s dispatch_due=%s active_pipeline_jobs=%s memory_available=%s",
        k8s_namespace,
        len(users),
        due_count,
        len(active_jobs),
        memory_available,
    )

    plan = plan_pipeline_dispatch(
        users,
        now=now,
        active_user_ids=active_user_ids,
        pipeline_job_active=bool(active_jobs),
        cooldown_blocked_user_ids=cooldown_blocked,
        memory_available=memory_available,
    )
    for user_id in plan.defer_user_ids:
        users_by_id[user_id].defer_pipeline_retry(db, now)
    if plan.defer_reason == DEFER_MEMORY:
        logger.info(
            "Deferring users until the next dispatcher tick: MemAvailable is below the training minimum. users=%s",
            ",".join(plan.defer_user_ids),
        )
    elif plan.defer_reason == DEFER_ACTIVE_JOB:
        logger.info(
            "Deferring users until the next dispatcher tick: a pipeline job is still running. users=%s",
            ",".join(plan.defer_user_ids),
        )
    elif plan.defer_reason == DEFER_WAITING_TURN:
        logger.info(
            "Deferred users to the next dispatcher tick so one pipeline starts at a time. users=%s",
            ",".join(plan.defer_user_ids),
        )

    spawned: list[str] = []
    if plan.spawn_user_id is not None:
        user = users_by_id[plan.spawn_user_id]
        queued = user.pipeline_retry_at is not None
        logger.info(
            "Dispatching pipeline job for user=%s cron_time=%s timezone=%s queued=%s",
            user.user_id,
            user.cron_time.strftime("%H:%M"),
            user.timezone,
            queued,
        )
        # Keep the retry if job creation fails, so the next tick tries again.
        user.defer_pipeline_retry(db, now)
        job_name = create_pipeline_job(
            namespace=k8s_namespace,
            user_id=user.user_id,
        )
        user.clear_pipeline_retry(db)
        spawned.append(job_name)

    result = DispatchResult(
        scanned_users=len(users),
        spawned_jobs=tuple(spawned),
        skipped_active_job=plan.skipped_active_job,
        skipped_not_due=plan.skipped_not_due,
        skipped_cooldown=plan.skipped_cooldown,
        deferred=len(plan.defer_user_ids),
    )
    logger.info(
        "Dispatch finished: users=%s spawned=%s skipped_active=%s skipped_not_due=%s skipped_cooldown=%s deferred=%s",
        result.scanned_users,
        len(result.spawned_jobs),
        result.skipped_active_job,
        result.skipped_not_due,
        result.skipped_cooldown,
        result.deferred,
    )
    return result


def main() -> int:
    from chess_teacher.utils.process_utils import log_script_runtime_context

    logger.info("Pipeline dispatcher started.")
    log_script_runtime_context(logger, script="dispatcher")
    result = dispatch_pipeline_jobs()
    if result.spawned_jobs:
        logger.info("Pipeline dispatcher spawned jobs: %s", ", ".join(result.spawned_jobs))
    logger.info(
        "Pipeline dispatcher completed: spawned=%s skipped_active=%s skipped_not_due=%s skipped_cooldown=%s deferred=%s",
        len(result.spawned_jobs),
        result.skipped_active_job,
        result.skipped_not_due,
        result.skipped_cooldown,
        result.deferred,
    )
    return 0


if __name__ == "__main__":
    from chess_teacher.utils.process_utils import run_script_main

    run_script_main(main)
