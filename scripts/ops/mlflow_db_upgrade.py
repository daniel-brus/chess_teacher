"""Upgrade the MLflow tracking database schema to match the installed package.

Same effect as ``mlflow db upgrade <url>``. Tracking URI defaults to
``MLFLOW_TRACKING_URI`` or the app Postgres URL (see ``MLflowTracker``).

Local::

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/ops/mlflow_db_upgrade.py --dry-run

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/ops/mlflow_db_upgrade.py

k3d / prod (whitelisted script Job; image must match the mlflow pin)::

    python scripts/utils/run_script_job.py mlflow_db_upgrade -- --dry-run
    python scripts/utils/run_script_job.py mlflow_db_upgrade
    /opt/chess_teacher/k8s/run-script-job.sh mlflow_db_upgrade -- --follow
"""

from __future__ import annotations

import argparse

from chess_teacher.pipelines.neural_network.mlflow_utils import (
    _log_tracking_uri,
    mlflow_tracking_schema_status,
    resolve_mlflow_tracking_uri,
    upgrade_mlflow_tracking_db,
)
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.process_utils import log_script_runtime_context, run_script_main

logger = get_logger()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print current vs head revision without migrating.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="Override MLFLOW_TRACKING_URI / default Postgres URL.",
    )
    args = parser.parse_args()
    log_script_runtime_context(logger, script="mlflow_db_upgrade")

    uri = resolve_mlflow_tracking_uri(args.tracking_uri)
    safe_uri = _log_tracking_uri(uri)
    current, head = mlflow_tracking_schema_status(uri)
    logger.info(
        "MLflow tracking schema status uri=%s current=%s head=%s",
        safe_uri,
        current,
        head,
    )
    print(f"tracking_uri={safe_uri}")
    print(f"current={current}")
    print(f"head={head}")

    if current == head and head is not None:
        print("already_at_head=true")
        logger.info("MLflow tracking schema already at head=%s", head)
        return 0

    if args.dry_run:
        print("dry_run=true would_upgrade=true")
        logger.info(
            "Dry-run only; would upgrade MLflow schema %s -> %s",
            current,
            head,
        )
        return 0

    before, after = upgrade_mlflow_tracking_db(uri)
    print(f"upgraded_from={before}")
    print(f"upgraded_to={after}")
    logger.info("MLflow tracking schema upgraded %s -> %s", before, after)
    if after != head:
        logger.error(
            "Upgrade finished but schema %s != package head %s",
            after,
            head,
        )
        return 1
    return 0


if __name__ == "__main__":
    run_script_main(main)
