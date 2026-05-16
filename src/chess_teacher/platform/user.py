from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from enum import StrEnum
from pathlib import Path
from typing import Any

from chess_teacher.utils.general_utils import assert_valid_timezone
from chess_teacher.utils.logging_utils import get_logger
from chess_teacher.utils.table_data_class import TableDataClass

DEFAULT_CRON_TIME = time(3, 0)
DEFAULT_TIMEZONE = "Europe/Amsterdam"

logger = get_logger()


class UserTier(StrEnum):
    FREE = "Free"
    PREMIUM = "Premium"


@dataclass()
class User(TableDataClass):
    """Represents an authenticated user."""

    user_id: str  # hashed unique ID
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
    def get_key(cls) -> str:
        return "users"

    @classmethod
    def get_yaml_path(cls) -> Path:
        return Path(__file__).parent / "metadata.yml"

    @classmethod
    def get_id_hash_columns(cls) -> tuple[str, ...]:
        return ("sub", "provider")

    @classmethod
    def get_timestamp_columns(cls) -> tuple[str, ...]:
        return ("latest_login", "latest_pipeline_run")

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
        """Create a User from a Streamlit authentication user object."""
        sub = st_user["sub"]
        provider = st_user["provider"]

        return cls(
            user_id=cls.generate_id({"sub": sub, "provider": provider}),
            sub=sub,
            provider=provider,
            email=st_user.get("email"),
            name=st_user.get("name"),
            picture=st_user.get("picture"),
            given_name=st_user.get("given_name"),
            family_name=st_user.get("family_name"),
            email_verified=st_user.get("email_verified", False),
            tier=tier,
            latest_login=latest_login,
            latest_pipeline_run=latest_pipeline_run,
            cron_time=cron_time,
            timezone=timezone,
        )
