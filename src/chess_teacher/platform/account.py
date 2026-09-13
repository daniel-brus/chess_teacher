from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self

from chess_teacher.utils.object_storage.images import asset_image_key
from chess_teacher.utils.table_data_class import TableDataClass

AppLogoVariant = Literal["black", "white"]

_CHESSCOM_USERNAME_RE = re.compile(r"^[a-z0-9_-]{3,25}$")
_LICHESS_USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,19}$")
_USERNAME_URL_HINT_RE = re.compile(r"[/.]|https?:", re.IGNORECASE)

CHESSCOM_USERNAME_MIN_LEN = 3
CHESSCOM_USERNAME_MAX_LEN = 25
LICHESS_USERNAME_MIN_LEN = 1
LICHESS_USERNAME_MAX_LEN = 20
PLATFORM_USERNAME_MAX_LEN = max(CHESSCOM_USERNAME_MAX_LEN, LICHESS_USERNAME_MAX_LEN)


def app_logo_key(*, variant: AppLogoVariant = "black") -> str:
    """Storage key for the Chess Teacher wordmark SVG (black or white)."""
    filename = "app-logo-white.svg" if variant == "white" else "app-logo-black.svg"
    return asset_image_key(filename)


Appearance = Literal["light", "dark"]


class AccountPlatform(StrEnum):
    CHESS_COM = "Chess.com"
    LICHESS = "Lichess"

    def logo_key(self, *, appearance: Appearance = "light") -> str:
        """Storage key for this platform's logo SVG under ``assets/images/``."""
        if self == AccountPlatform.CHESS_COM:
            return asset_image_key("chesscom_logo_pawn.svg")
        if self == AccountPlatform.LICHESS:
            filename = "lichess-white.svg" if appearance == "dark" else "lichess.svg"
            return asset_image_key(filename)
        raise ValueError(f"Unknown platform: {self}")


def normalize_platform_username(username: str, platform: AccountPlatform) -> str:
    """Strip, lowercase, and validate a Chess.com or Lichess username."""
    cleaned = username.strip().lstrip("@").strip()
    if not cleaned:
        raise ValueError("Enter a username.")
    if _USERNAME_URL_HINT_RE.search(cleaned):
        raise ValueError("Enter the username, not a profile URL.")

    normalized = cleaned.lower()
    if platform is AccountPlatform.CHESS_COM:
        if not _CHESSCOM_USERNAME_RE.fullmatch(normalized):
            raise ValueError(
                "Chess.com usernames are 3-25 letters, numbers, underscores, or hyphens."
            )
        return normalized
    if platform is AccountPlatform.LICHESS:
        if not _LICHESS_USERNAME_RE.fullmatch(normalized):
            raise ValueError(
                "Lichess usernames are 1-20 letters, numbers, underscores, or hyphens."
            )
        return normalized
    raise ValueError(f"Unknown platform: {platform}")


@dataclass()
class Account(TableDataClass):
    """Represents an account on a chess platform."""

    account_id: str  # hashed unique ID
    username: str
    platform: AccountPlatform
    latest_ingestion: datetime | None = None

    @classmethod
    def from_username_and_platform(cls, username: str, platform: AccountPlatform) -> Account:
        username = normalize_platform_username(username, platform)
        return cls(
            account_id=cls.generate_id({"username": username, "platform": platform}),
            username=username,
            platform=platform,
        )

    def to_cache_dict(self) -> dict[str, Any]:
        """Serialize for Redis / JSON caches (platform layer owns Account shape)."""
        return {
            "account_id": self.account_id,
            "username": self.username,
            "platform": self.platform.value,
            "latest_ingestion": (
                self.latest_ingestion.isoformat() if self.latest_ingestion is not None else None
            ),
        }

    @classmethod
    def from_cache_dict(cls, data: dict[str, Any]) -> Self:
        """Restore from ``to_cache_dict`` payload."""
        latest_ingestion = data.get("latest_ingestion")
        return cls(
            account_id=data["account_id"],
            username=data["username"],
            platform=AccountPlatform(data["platform"]),
            latest_ingestion=(
                datetime.fromisoformat(latest_ingestion) if latest_ingestion is not None else None
            ),
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
