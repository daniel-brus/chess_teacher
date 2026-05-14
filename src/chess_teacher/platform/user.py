from dataclasses import asdict, dataclass
from datetime import datetime

from chess_teacher.utils.db_client import DatabaseClient
from chess_teacher.utils.logging_utils import get_logger
from chess_teacher.utils.metadata_utils import TableMetadata

logger = get_logger()


@dataclass
class User:
    """Represents an authenticated user."""

    id: str  # hashed unique ID
    sub: str  # unique ID per provider, no fallback
    email: str | None = None
    name: str | None = None
    picture: str | None = None
    given_name: str | None = None
    family_name: str | None = None
    provider: str | None = None
    email_verified: bool = False
    latest_login: datetime | None = None
    latest_pipeline_run: datetime | None = None

    def save_to_db(self, db_client: DatabaseClient):
        """Insert the user into the database, if not already exists."""
        try:
            tablemetadata = TableMetadata("users")
            db_client.ensure_table(tablemetadata)
            result = db_client.insert([asdict(self)], tablemetadata, on_conflict="nothing")
            if result.rows_inserted == 1:
                logger.info(f"User {self.email} saved to database (ID: {self.id}).")
        except Exception as e:
            logger.log_and_raise(e)


def get_test_user():
    return User(id="test", sub="test")
