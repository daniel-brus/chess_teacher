# Remove orphaned accounts (no link to existing users)

# Remove orphaned pipeline runs (finished_at EPOCH, started long enough ago)

from chess_teacher.maintenance.main import run_maintenance
from chess_teacher.utils.logging import get_logger

logger = get_logger()


def main() -> int:
    logger.info("Maintenance job started: Maintenance pipeline.")
    run_maintenance()
    logger.info("Maintenance job completed.")
    return 0


if __name__ == "__main__":
    from chess_teacher.utils.process_utils import run_script_main

    run_script_main(main)
