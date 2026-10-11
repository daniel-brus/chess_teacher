from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from chess_teacher.platform.user import User
from chess_teacher.utils.db.client import DatabaseClient, get_db_client
from chess_teacher.utils.env_utils import get_env_variable
from chess_teacher.utils.logging import get_logger
from orchestration.pipeline_jobs import (
    create_pipeline_job,
    list_active_pipeline_jobs,
    work_item_key,
)

logger = get_logger()


@dataclass(frozen=True)
class DispatchResult:
    scanned_users: int
    spawned_jobs: tuple[str, ...]
    skipped_active_job: int
    skipped_not_due: int
    skipped_cooldown: int


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
    cron_due_count = sum(1 for user in users if user.is_cron_due(now))
    logger.info(
        "Dispatch scan: namespace=%s users=%s cron_due=%s active_pipeline_jobs=%s",
        k8s_namespace,
        len(users),
        cron_due_count,
        len(active_jobs),
    )

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

        logger.info(
            "Dispatching pipeline job for user=%s cron_time=%s timezone=%s",
            user.user_id,
            user.cron_time.strftime("%H:%M"),
            user.timezone,
        )
        job_name = create_pipeline_job(namespace=k8s_namespace, user_id=user.user_id)
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


def main() -> int:
    from chess_teacher.utils.process_utils import log_script_runtime_context

    logger.info("Pipeline dispatcher started.")
    log_script_runtime_context(logger, script="dispatcher")
    result = dispatch_pipeline_jobs()
    if result.spawned_jobs:
        logger.info("Pipeline dispatcher spawned jobs: %s", ", ".join(result.spawned_jobs))
    logger.info(
        "Pipeline dispatcher completed: spawned=%s skipped_active=%s skipped_not_due=%s skipped_cooldown=%s",
        len(result.spawned_jobs),
        result.skipped_active_job,
        result.skipped_not_due,
        result.skipped_cooldown,
    )
    return 0


if __name__ == "__main__":
    from chess_teacher.utils.process_utils import run_script_main

    run_script_main(main)
