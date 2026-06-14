from __future__ import annotations

import argparse

from chess_teacher.platform.user import User
from chess_teacher.runner import latest_successful_run_id, run_pipeline
from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.logging import get_logger

logger = get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pipelines for one user.")
    parser.add_argument("--user-id", required=True)
    args = parser.parse_args()

    db_client = get_db_client()
    user = User.fetch_from_db(db_client, id=args.user_id)

    logger.info("Pipeline job started for user=%s.", user.user_id)
    results = run_pipeline(user, db_client)

    run_id = latest_successful_run_id(results)
    if run_id is not None:
        user.update_latest_pipeline_run(db_client, run_id)

    logger.info(
        "Pipeline job finished for user=%s: accounts=%s results=%s.",
        user.user_id,
        len(results),
        [result.result.value for result in results],
    )


if __name__ == "__main__":
    main()
