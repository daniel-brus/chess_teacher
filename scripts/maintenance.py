# Remove orphaned accounts (no link to existing users)

# Remove orphaned pipeline runs (finished_at EPOCH, started long enough ago)

from chess_teacher.maintenance.main import run_maintenance
from chess_teacher.utils.logging_utils import get_logger

logger = get_logger()


def main():
    logger.info("Maintenance job started: Maintenance pipeline.")
    run_maintenance()
    logger.info("Maintenance job completed.")


if __name__ == "__main__":
    main()
