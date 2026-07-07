"""Redis cache helpers for user-scoped read paths (TCP via redis-py)."""

from __future__ import annotations

import io
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import polars as pl

from chess_teacher.platform.account import Account, AccountPlatform
from chess_teacher.utils.logging import get_logger

logger = get_logger()

CACHE_KEY_VERSION = "v1"
USER_GAMES_TTL_SECONDS = 3600
USER_ACCOUNTS_TTL_SECONDS = 1800

_UNSET = object()
_cache_client: CacheClient | None | object = _UNSET


def user_games_cache_key(user_id: str) -> str:
    return f"user:{user_id}:games:{CACHE_KEY_VERSION}"


def user_accounts_cache_key(user_id: str) -> str:
    return f"user:{user_id}:accounts:{CACHE_KEY_VERSION}"


def _account_to_cache_dict(account: Account) -> dict[str, Any]:
    return {
        "account_id": account.account_id,
        "username": account.username,
        "platform": account.platform.value,
        "latest_ingestion": (
            account.latest_ingestion.isoformat() if account.latest_ingestion is not None else None
        ),
    }


def _account_from_cache_dict(data: dict[str, Any]) -> Account:
    latest_ingestion = data.get("latest_ingestion")
    return Account(
        account_id=data["account_id"],
        username=data["username"],
        platform=AccountPlatform(data["platform"]),
        latest_ingestion=(
            datetime.fromisoformat(latest_ingestion) if latest_ingestion is not None else None
        ),
    )


def _encode_polars(df: pl.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.write_parquet(buffer)
    return buffer.getvalue()


def _decode_polars(data: bytes) -> pl.DataFrame:
    return pl.read_parquet(io.BytesIO(data))


class CacheClient(ABC):
    @abstractmethod
    def get_user_games(self, user_id: str) -> pl.DataFrame | None: ...

    @abstractmethod
    def set_user_games(self, user_id: str, games: pl.DataFrame) -> None: ...

    @abstractmethod
    def get_user_accounts(self, user_id: str) -> list[Account] | None: ...

    @abstractmethod
    def set_user_accounts(self, user_id: str, accounts: Sequence[Account]) -> None: ...

    @abstractmethod
    def delete(self, *keys: str) -> None: ...


class RedisCacheClient(CacheClient):
    def __init__(self, client: Any) -> None:
        self._client = client

    def get_user_games(self, user_id: str) -> pl.DataFrame | None:
        return self._get_polars(user_games_cache_key(user_id))

    def set_user_games(self, user_id: str, games: pl.DataFrame) -> None:
        self._set_polars(user_games_cache_key(user_id), games, USER_GAMES_TTL_SECONDS)

    def get_user_accounts(self, user_id: str) -> list[Account] | None:
        payload = self._get_json(user_accounts_cache_key(user_id))
        if payload is None:
            return None
        return [_account_from_cache_dict(item) for item in payload]

    def set_user_accounts(self, user_id: str, accounts: Sequence[Account]) -> None:
        payload = [_account_to_cache_dict(account) for account in accounts]
        self._set_json(user_accounts_cache_key(user_id), payload, USER_ACCOUNTS_TTL_SECONDS)

    def delete(self, *keys: str) -> None:
        if not keys:
            return
        try:
            self._client.delete(*keys)
        except Exception:
            logger.warning("Failed to delete Redis keys: %s", keys, exc_info=True)

    def _get_json(self, key: str) -> list[dict[str, Any]] | None:
        try:
            raw = self._client.get(key)
        except Exception:
            logger.warning("Redis GET failed for key=%s", key, exc_info=True)
            return None
        if raw is None:
            return None
        return json.loads(raw.decode("utf-8"))

    def _set_json(self, key: str, value: list[dict[str, Any]], ttl_seconds: int) -> None:
        try:
            self._client.set(key, json.dumps(value).encode("utf-8"), ex=ttl_seconds)
        except Exception:
            logger.warning("Redis SET failed for key=%s", key, exc_info=True)

    def _get_polars(self, key: str) -> pl.DataFrame | None:
        try:
            raw = self._client.get(key)
        except Exception:
            logger.warning("Redis GET failed for key=%s", key, exc_info=True)
            return None
        if raw is None:
            return None
        return _decode_polars(raw)

    def _set_polars(self, key: str, df: pl.DataFrame, ttl_seconds: int) -> None:
        try:
            self._client.set(key, _encode_polars(df), ex=ttl_seconds)
        except Exception:
            logger.warning("Redis SET failed for key=%s", key, exc_info=True)


def get_cache_client() -> CacheClient | None:
    """Return a shared cache client, or None when Redis is unavailable or unset."""
    global _cache_client
    if _cache_client is not _UNSET:
        return _cache_client  # type: ignore[return-value]

    redis_url = os.getenv("REDIS_URL", "").strip()
    if not redis_url:
        _cache_client = None
        return None

    try:
        import redis

        client = redis.from_url(redis_url, decode_responses=False)
        client.ping()
        _cache_client = RedisCacheClient(client)
        logger.info("Redis cache enabled.")
    except Exception:
        logger.warning("Redis unavailable; caching disabled.", exc_info=True)
        _cache_client = None
    return _cache_client  # type: ignore[return-value]


def invalidate_user_games_cache(user_id: str) -> None:
    cache = get_cache_client()
    if cache is not None:
        cache.delete(user_games_cache_key(user_id))


def invalidate_user_accounts_cache(user_id: str) -> None:
    cache = get_cache_client()
    if cache is not None:
        cache.delete(user_accounts_cache_key(user_id))


def invalidate_user_games_and_accounts_cache(user_id: str) -> None:
    cache = get_cache_client()
    if cache is not None:
        cache.delete(user_games_cache_key(user_id), user_accounts_cache_key(user_id))


def reset_cache_client_for_tests() -> None:
    """Clear the module-level singleton (tests only)."""
    global _cache_client
    _cache_client = _UNSET
