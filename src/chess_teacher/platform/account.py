from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from chess_teacher.utils.env_utils import get_env_variable
from chess_teacher.utils.exception_utils import ConfigError
from chess_teacher.utils.table_data_class import TableDataClass


def platform_logo_images_dir() -> Path:
    """``{RAW_DIR}/assets/images`` — platform SVG logos live here."""
    raw_dir = get_env_variable("RAW_DIR")
    if not raw_dir:
        raise ConfigError("RAW_DIR environment variable is not set")
    return Path(raw_dir) / "assets" / "images"


AppLogoVariant = Literal["black", "white"]


def app_logo_path(*, variant: AppLogoVariant = "black") -> Path:
    """Chess Teacher wordmark SVG (black for light UI, white for dark UI)."""
    filename = "app-logo-white.svg" if variant == "white" else "app-logo-black.svg"
    return platform_logo_images_dir() / filename


Appearance = Literal["light", "dark"]


class AccountPlatform(StrEnum):
    CHESS_COM = "Chess.com"
    LICHESS = "Lichess"

    def logo_path(self, *, appearance: Appearance = "light") -> Path:
        if self == AccountPlatform.CHESS_COM:
            return platform_logo_images_dir() / "chesscom_logo_pawn.svg"
        if self == AccountPlatform.LICHESS:
            filename = "lichess-white.svg" if appearance == "dark" else "lichess.svg"
            return platform_logo_images_dir() / filename
        raise ValueError(f"Unknown platform: {self}")


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

    def format_label(self) -> str:
        return f"{self.platform.value} · {self.username}"
