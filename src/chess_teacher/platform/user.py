from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, fields
from datetime import UTC, datetime, time
from enum import Enum
from typing import Any

from chess_teacher.utils.db_client import DatabaseClient
from chess_teacher.utils.exception_utils import ConfigError, DatabaseError
from chess_teacher.utils.general_utils import (
    assert_valid_timezone,
    generate_hash,
    generate_ident_is_literal,
)
from chess_teacher.utils.logging_utils import get_logger
from chess_teacher.utils.metadata_utils import TableMetadata

logger = get_logger()


class UserTier(Enum):
    FREE = "Free"
    PREMIUM = "Premium"


DEFAULT_CRON_TIME = time(3, 0)
DEFAULT_TIMEZONE = "Europe/Amsterdam"


@dataclass()
class User:
    """Represents an authenticated user."""

    """Represents an authenticated user."""

    id: str  # hashed unique ID
    sub: str
    provider: str
    email: str | None = None
    name: str | None = None
    picture: str | None = None
    given_name: str | None = None
    family_name: str | None = None
    email_verified: bool = False
    tier: UserTier = UserTier.FREE
    latest_login: datetime | None = None
    latest_pipeline_run: datetime | None = None
    cron_time: time = DEFAULT_CRON_TIME
    timezone: str = DEFAULT_TIMEZONE

    def __post_init__(self) -> None:
        assert_valid_timezone(self.timezone)

    @classmethod
    def from_st_user(
        cls,
        st_user: dict[str, Any],
        *,
        tier: UserTier = UserTier.FREE,
        latest_login: datetime | None = None,
        latest_pipeline_run: datetime | None = None,
        cron_time: time = DEFAULT_CRON_TIME,
        timezone: str = DEFAULT_TIMEZONE,
    ) -> User:
        """
        Create a User from a Streamlit authentication user object.
        """

        sub = st_user["sub"]
        provider = st_user["provider"]

        return cls(
            id=generate_hash([sub, provider]),
            sub=sub,
            provider=provider,
            email=st_user.get("email"),
            name=st_user.get("name"),
            picture=st_user.get("picture"),
            given_name=st_user.get("given_name"),
            family_name=st_user.get("family_name"),
            email_verified=st_user.get(
                "email_verified",
                False,
            ),
            tier=tier,
            latest_login=latest_login,
            latest_pipeline_run=latest_pipeline_run,
            cron_time=cron_time,
            timezone=timezone,
        )

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> User:
        """
        Create a User from a fully materialized dictionary.

        Required:
        - all fields without defaults must exist
        Optional:
        - missing fields with defaults fall back automatically
        """

        kwargs = {}

        for field in fields(cls):
            field_name = field.name

            if field_name in data:
                kwargs[field_name] = data[field_name]

            elif field.default is not MISSING:
                kwargs[field_name] = field.default

            elif field.default_factory is not MISSING:
                kwargs[field_name] = field.default_factory()

            else:
                ValueError(f'Missing required field "{field_name}" for User.from_dict().')

        return cls(**kwargs)

    @staticmethod
    def generate_id(user: dict = {}) -> str:
        try:
            user_id = generate_hash([user.get("sub"), user.get("provider")])
            return user_id
        except Exception:
            logger.log_and_raise(
                ConfigError(f"sub/provider combination can not be obtained from user: {user}")
            )

    @staticmethod
    def fetch_from_db(db_client: DatabaseClient, *, id: str | None = None, user: dict = {}) -> User:
        user_id = id
        if not user_id:  # derive id from st.user
            user_id = User.generate_id(user)

        try:
            tablemetadata = TableMetadata("users")
            where = generate_ident_is_literal("id", user_id)
            result = db_client.read(tablemetadata, where=where)
            if len(result) != 1:
                logger.log_and_raise(
                    DatabaseError(
                        f"Could not found unique user ({len(result)} results) in DB with {where}"
                    )
                )
        except Exception as e:
            logger.log_and_raise(e)
        u = result[0]
        return User.from_dict(u)

    @staticmethod
    def exists_in_db(db_client: DatabaseClient, id: str) -> bool:
        """Returns True if precisely 1 row matches the id, False if 0. Raises Exception otherwise."""
        try:
            tablemetadata = TableMetadata("users")
            return db_client.exists(tablemetadata, where=generate_ident_is_literal("id", id))
        except Exception as e:
            logger.log_and_raise(e)

    def save_to_db(self, db_client: DatabaseClient) -> None:
        """Insert the user into the database, if not already exists."""
        try:
            tablemetadata = TableMetadata("users")
            db_client.ensure_table(tablemetadata)
            result = db_client.insert([asdict(self)], tablemetadata, on_conflict="nothing")
            if result.rows_inserted == 1:
                logger.info(f"User {self.email} saved to database (ID: {self.id}).")
        except Exception as e:
            logger.log_and_raise(e)

    def upsert_field(self, db_client: DatabaseClient, field: str, value) -> None:
        try:
            tablemetadata = TableMetadata("users")
            db_client.update_where(
                tablemetadata, {field: value}, where=generate_ident_is_literal("id", self.id)
            )
        except Exception as e:
            logger.log_and_raise(e)

    def upsert_latest(self, db_client, field: str, ts: datetime = datetime.now(UTC)) -> None:
        allowed_fields = ["latest_login", "latest_pipeline_run"]
        if field not in allowed_fields:
            logger.log_and_raise(
                ConfigError(
                    f"Error: trying to upsert illegal field ({field}) must be in {", ".join(allowed_fields)}."
                )
            )
        self.upsert_field(db_client, field, ts)

    def delete_from_db(self, db_client: DatabaseClient) -> None:
        try:
            tablemetadata = TableMetadata("users")
            db_client.delete_where(tablemetadata, where=generate_ident_is_literal("id", self.id))
        except Exception as e:
            logger.log_and_raise(e)

    @staticmethod
    def get_metadata() -> TableMetadata:
        try:
            return TableMetadata("users")
        except Exception as e:
            logger.log_and_raise(e)
