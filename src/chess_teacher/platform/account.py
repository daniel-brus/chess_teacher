from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from chess_teacher.utils.table_data_class import TableDataClass


class AccountPlatform(StrEnum):
    CHESS_COM = "Chess.com"
    LICHESS = "Lichess"


@dataclass()
class Account(TableDataClass):
    """Represents an account on a chess platform."""

    account_id: str  # hashed unique ID
    username: str
    platform: AccountPlatform
    latest_ingestion: datetime | None = None

    @classmethod
    def from_username_and_platform(cls, username: str, platform: AccountPlatform) -> Account:
        return cls(
            account_id=cls.generate_id({"username": username, "platform": platform}),
            username=username,
            platform=platform,
        )

    @classmethod
    def get_key(cls) -> str:
        return "accounts"

    @classmethod
    def get_yaml_path(cls) -> Path:
        return Path(__file__).parent / "metadata.yml"

    @classmethod
    def get_id_hash_columns(cls) -> tuple[str, ...]:
        return ("username", "platform")

    @classmethod
    def get_timestamp_columns(cls) -> tuple[str, ...]:
        return ("latest_ingestion",)
