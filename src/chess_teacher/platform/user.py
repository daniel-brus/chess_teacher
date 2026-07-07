from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, BinaryIO, Self
from zoneinfo import ZoneInfo

from chess_teacher.platform.account import Account, AppLogoVariant
from chess_teacher.platform.profile_picture import clear_upload_image_cache, profile_pictures
from chess_teacher.platform.user_account import UserAccount
from chess_teacher.utils.cache_utils import (
    get_cache_client,
    invalidate_user_games_and_accounts_cache,
)
from chess_teacher.utils.db.client import DatabaseClient
from chess_teacher.utils.exception_utils import DatabaseError
from chess_teacher.utils.general_utils import (
    assert_valid_timezone,
    generate_ident_is_literal,
    get_current_datetime,
)
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.pipeline_utils.pipeline_helpers import PipelineRunResult
from chess_teacher.utils.table_data_class import TableDataClass

DEFAULT_CRON_TIME = time(3, 0)
DEFAULT_TIMEZONE = "Europe/Amsterdam"
DISPATCH_INTERVAL_MINUTES = 30
DISPATCH_INTERVAL = timedelta(minutes=DISPATCH_INTERVAL_MINUTES)
PIPELINE_RUN_COOLDOWN = timedelta(hours=24)
# Aligns with dispatcher poll interval; avoids blocking the next day's slot when
# yesterday's run started slightly after the nominal cron time.
PIPELINE_RUN_COOLDOWN_MARGIN = DISPATCH_INTERVAL

logger = get_logger()


def dispatch_cron_time_options() -> tuple[time, ...]:
    """Daily run times aligned with the ingestion dispatcher's 30-minute poll interval."""
    return tuple(time(hour, minute) for hour in range(24) for minute in (0, 30))


def format_cron_time_label(value: time) -> str:
    return value.strftime("%H:%M")


def normalize_cron_time(value: time) -> time:
    """Snap a time down to the nearest allowed dispatch interval."""
    total_minutes = value.hour * 60 + value.minute
    snapped = (total_minutes // DISPATCH_INTERVAL_MINUTES) * DISPATCH_INTERVAL_MINUTES
    return time(snapped // 60, snapped % 60)


def cron_time_option_index(value: time) -> int:
    options = dispatch_cron_time_options()
    normalized = normalize_cron_time(value)
    try:
        return options.index(normalized)
    except ValueError:
        return options.index(DEFAULT_CRON_TIME)


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
    latest_pipeline_run: str | None = None
    cron_time: time = DEFAULT_CRON_TIME
    timezone: str = DEFAULT_TIMEZONE
    default_light_theme_id: str | None = None
    default_dark_theme_id: str | None = None

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
        return ("latest_login",)

    @classmethod
    def from_st_user(
        cls,
        st_user: dict[str, Any],
        *,
        tier: UserTier = UserTier.FREE,
        latest_login: datetime | None = None,
        latest_pipeline_run: str | None = None,
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

    def replace_profile_picture(
        self,
        db_client: DatabaseClient,
        *,
        data: bytes | BinaryIO,
        original_filename: str,
    ) -> Self:
        """Replace this user's avatar with an uploaded image; persist to storage and DB."""
        upload_bytes = data if isinstance(data, bytes) else data.read()
        clear_upload_image_cache(self.picture)
        profile_pictures.delete(self.picture)
        picture_url = profile_pictures.save(
            user_id=self.user_id,
            data=upload_bytes,
            original_filename=original_filename,
        )
        clear_upload_image_cache(picture_url)
        self.upsert_field(db_client, "picture", picture_url)
        self.picture = picture_url
        return self

    def replace_profile_picture_with_app_logo(
        self,
        db_client: DatabaseClient,
        *,
        variant: AppLogoVariant,
    ) -> Self:
        """Point this user's avatar at a bundled black/white wordmark (no copy into uploads)."""
        clear_upload_image_cache(self.picture)
        profile_pictures.delete(self.picture)
        picture_ref = profile_pictures.app_logo_picture_ref(variant=variant)
        self.upsert_field(db_client, "picture", picture_ref)
        self.picture = picture_ref
        return self

    def update_name(self, db_client: DatabaseClient, name: str | None) -> Self:
        """Update this user's display name."""
        self.upsert_field(db_client, "name", name)
        self.name = name
        return self

    def update_cron_time(self, db_client: DatabaseClient, cron_time: time) -> Self:
        """Update the daily cron time (interpreted in this user's timezone)."""
        cron_time = normalize_cron_time(cron_time)
        self.upsert_field(db_client, "cron_time", cron_time)
        self.cron_time = cron_time
        return self

    def update_timezone(self, db_client: DatabaseClient, timezone: str) -> Self:
        """Update the timezone used for cron_time."""
        assert_valid_timezone(timezone)
        self.upsert_field(db_client, "timezone", timezone)
        self.timezone = timezone
        return self

    def update_latest_pipeline_run(self, db_client: DatabaseClient, run_id: str) -> Self:
        """Point this user at the given pipeline run as their most recent run."""
        self.upsert_field(db_client, "latest_pipeline_run", run_id)
        self.latest_pipeline_run = run_id
        return self

    def get_latest_pipeline_run(self, db_client: DatabaseClient) -> PipelineRunResult | None:
        """Fetch the pipeline run referenced by latest_pipeline_run, if any."""
        if self.latest_pipeline_run is None:
            return None
        try:
            return PipelineRunResult.fetch_from_db(db_client, id=self.latest_pipeline_run)
        except DatabaseError:
            logger.warning(
                "User %s references missing pipeline run %s.",
                self.user_id,
                self.latest_pipeline_run,
            )
            return None

    def is_cron_due(self, now: datetime | None = None) -> bool:
        """True when ``now`` falls in today's cron window in this user's timezone.

        The window is ``[cron_time, cron_time + DISPATCH_INTERVAL)``, matching the
        ingestion dispatcher's 30-minute poll cadence.
        """
        current = now or get_current_datetime()
        local_now = current.astimezone(ZoneInfo(self.timezone))
        scheduled = local_now.replace(
            hour=self.cron_time.hour,
            minute=self.cron_time.minute,
            second=0,
            microsecond=0,
        )
        if local_now < scheduled:
            return False
        return local_now < scheduled + DISPATCH_INTERVAL

    def pipeline_allowed_to_run(self, db_client: DatabaseClient) -> bool:
        """Check if the pipeline is allowed to run for this user.

        Cooldown is measured from when the latest run *started*, not when it finished,
        so a long run does not push the next daily cron slot past the 24h window.
        A small margin aligns with the ingestion dispatcher's 30-minute poll interval.
        """
        # TODO: User tier logic
        latest_run = self.get_latest_pipeline_run(db_client)
        if latest_run is None:
            return True
        earliest_next = latest_run.started_at + PIPELINE_RUN_COOLDOWN - PIPELINE_RUN_COOLDOWN_MARGIN
        return get_current_datetime() >= earliest_next

    def get_linked_accounts(self, db_client: DatabaseClient) -> list[Account]:
        """Fetch platform accounts linked to this user via the bridge table."""
        cache = get_cache_client()
        if cache is not None:
            cached_accounts = cache.get_user_accounts(self.user_id)
            if cached_accounts is not None:
                return cached_accounts

        accounts = self._load_linked_accounts_from_db(db_client)
        logger.info(
            "Loaded linked accounts from database user_id=%s count=%s",
            self.user_id,
            len(accounts),
        )

        if cache is not None:
            cache.set_user_accounts(self.user_id, accounts)

        return accounts

    def _load_linked_accounts_from_db(self, db_client: DatabaseClient) -> list[Account]:
        db_client.ensure_table(Account.get_metadata())

        br_metadata = UserAccount.get_metadata()
        db_client.ensure_table(br_metadata)

        user_accounts = db_client.read(
            br_metadata,
            where=generate_ident_is_literal("user_id", self.user_id),
            order_by="account_id",
        )
        accounts: list[Account] = []
        for user_account in user_accounts:
            try:
                accounts.append(Account.fetch_from_db(db_client, id=user_account["account_id"]))
            except DatabaseError:
                logger.warning(
                    "Removing stale account link for user %s and account %s",
                    self.user_id,
                    user_account["account_id"],
                )
                UserAccount.from_dict(user_account).delete_from_db(db_client)
        return accounts

    def unlink_all_accounts(self, db_client: DatabaseClient) -> None:
        """Delete all bridge rows for this user. Platform account rows are kept."""
        br_metadata = UserAccount.get_metadata()
        db_client.ensure_table(br_metadata)

        user_accounts = db_client.read(
            br_metadata,
            where=generate_ident_is_literal("user_id", self.user_id),
        )
        for user_account in user_accounts:
            UserAccount.from_dict(user_account).delete_from_db(db_client)
        invalidate_user_games_and_accounts_cache(self.user_id)

    def link_account(self, db_client: DatabaseClient, account: Account) -> bool:
        """Persist the account and link it to this user. Returns False if already linked."""
        account.save_new_to_db(db_client)
        user_account = UserAccount(
            user_id=self.user_id,
            account_id=account.account_id,
        )
        linked = user_account.save_new_to_db(db_client)
        if linked:
            invalidate_user_games_and_accounts_cache(self.user_id)
        return linked

    def unlink_account(self, db_client: DatabaseClient, account: Account) -> bool:
        """Delete the bridge row for this user and account. The account row is kept."""
        user_account = UserAccount(
            user_id=self.user_id,
            account_id=account.account_id,
        )
        if not db_client.exists(UserAccount.get_metadata(), user_account.get_where_clause()):
            logger.log_and_raise(
                DatabaseError(f"Account {account.account_id} not linked to user {self.user_id}")
            )
        user_account.delete_from_db(db_client)
        invalidate_user_games_and_accounts_cache(self.user_id)
        return True
