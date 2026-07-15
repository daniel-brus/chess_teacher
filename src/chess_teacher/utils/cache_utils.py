"""Redis cache helpers for user-scoped read paths (TCP via redis-py)."""

from __future__ import annotations

import io
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import polars as pl

from chess_teacher.platform.account import Account, AccountPlatform
from chess_teacher.utils.logging import get_logger

logger = get_logger()

CACHE_KEY_VERSION = "v1"
USER_GAMES_TTL_SECONDS = 3600
USER_ACCOUNTS_TTL_SECONDS = 1800
ADMIN_LOG_AGGREGATES_TTL_SECONDS = 3600

_UNSET = object()
_cache_client: CacheClient | None | object = _UNSET


def user_games_cache_key(user_id: str) -> str:
    return f"user:{user_id}:games:{CACHE_KEY_VERSION}"


def user_accounts_cache_key(user_id: str) -> str:
    return f"user:{user_id}:accounts:{CACHE_KEY_VERSION}"


def log_level_hourly_counts_cache_key() -> str:
    return f"admin:log_level_hourly_counts:{CACHE_KEY_VERSION}"


def exception_hourly_counts_cache_key() -> str:
    return f"admin:exception_hourly_counts:{CACHE_KEY_VERSION}"


def _redis_endpoint_label(redis_url: str) -> str:
    """Return a log-safe Redis endpoint label (host:port, no credentials)."""
    parsed = urlparse(redis_url)
    host = parsed.hostname or "unknown-host"
    if parsed.port is not None:
        return f"{host}:{parsed.port}"
    return host


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
    def get_log_level_hourly_counts(self) -> pl.DataFrame | None: ...

    @abstractmethod
    def set_log_level_hourly_counts(self, counts: pl.DataFrame) -> None: ...

    @abstractmethod
    def get_exception_hourly_counts(self) -> pl.DataFrame | None: ...

    @abstractmethod
    def set_exception_hourly_counts(self, counts: pl.DataFrame) -> None: ...

    @abstractmethod
    def delete(self, *keys: str) -> None: ...


class RedisCacheClient(CacheClient):
    def __init__(self, client: Any, *, endpoint: str) -> None:
        self._client = client
        self._endpoint = endpoint

    def get_user_games(self, user_id: str) -> pl.DataFrame | None:
        key = user_games_cache_key(user_id)
        games = self._get_polars(key)
        if games is None:
            logger.info("Redis user games cache miss user_id=%s key=%s", user_id, key)
            return None
        logger.info(
            "Redis user games cache hit user_id=%s key=%s rows=%s",
            user_id,
            key,
            games.height,
        )
        return games

    def set_user_games(self, user_id: str, games: pl.DataFrame) -> None:
        key = user_games_cache_key(user_id)
        self._set_polars(key, games, USER_GAMES_TTL_SECONDS)
        logger.debug(
            "Redis user games cache populated user_id=%s key=%s rows=%s ttl=%ss",
            user_id,
            key,
            games.height,
            USER_GAMES_TTL_SECONDS,
        )

    def get_user_accounts(self, user_id: str) -> list[Account] | None:
        key = user_accounts_cache_key(user_id)
        payload = self._get_json(key)
        if payload is None:
            logger.info("Redis user accounts cache miss user_id=%s key=%s", user_id, key)
            return None
        accounts = [_account_from_cache_dict(item) for item in payload]
        logger.info(
            "Redis user accounts cache hit user_id=%s key=%s count=%s",
            user_id,
            key,
            len(accounts),
        )
        return accounts

    def set_user_accounts(self, user_id: str, accounts: Sequence[Account]) -> None:
        key = user_accounts_cache_key(user_id)
        payload = [_account_to_cache_dict(account) for account in accounts]
        self._set_json(key, payload, USER_ACCOUNTS_TTL_SECONDS)
        logger.debug(
            "Redis user accounts cache populated user_id=%s key=%s count=%s ttl=%ss",
            user_id,
            key,
            len(payload),
            USER_ACCOUNTS_TTL_SECONDS,
        )

    def get_log_level_hourly_counts(self) -> pl.DataFrame | None:
        key = log_level_hourly_counts_cache_key()
        counts = self._get_polars(key)
        if counts is None:
            logger.info("Redis log level hourly counts cache miss key=%s", key)
            return None
        logger.info(
            "Redis log level hourly counts cache hit key=%s rows=%s",
            key,
            counts.height,
        )
        return counts

    def set_log_level_hourly_counts(self, counts: pl.DataFrame) -> None:
        key = log_level_hourly_counts_cache_key()
        self._set_polars(key, counts, ADMIN_LOG_AGGREGATES_TTL_SECONDS)
        logger.debug(
            "Redis log level hourly counts cache populated key=%s rows=%s ttl=%ss",
            key,
            counts.height,
            ADMIN_LOG_AGGREGATES_TTL_SECONDS,
        )

    def get_exception_hourly_counts(self) -> pl.DataFrame | None:
        key = exception_hourly_counts_cache_key()
        counts = self._get_polars(key)
        if counts is None:
            logger.info("Redis exception hourly counts cache miss key=%s", key)
            return None
        logger.info(
            "Redis exception hourly counts cache hit key=%s rows=%s",
            key,
            counts.height,
        )
        return counts

    def set_exception_hourly_counts(self, counts: pl.DataFrame) -> None:
        key = exception_hourly_counts_cache_key()
        self._set_polars(key, counts, ADMIN_LOG_AGGREGATES_TTL_SECONDS)
        logger.debug(
            "Redis exception hourly counts cache populated key=%s rows=%s ttl=%ss",
            key,
            counts.height,
            ADMIN_LOG_AGGREGATES_TTL_SECONDS,
        )

    def delete(self, *keys: str) -> None:
        if not keys:
            return
        try:
            deleted = self._client.delete(*keys)
            logger.info(
                "Redis cache invalidated endpoint=%s keys=%s deleted=%s",
                self._endpoint,
                list(keys),
                deleted,
            )
        except Exception:
            logger.warning(
                "Redis DELETE failed endpoint=%s keys=%s",
                self._endpoint,
                list(keys),
                exc_info=True,
            )

    def _get_json(self, key: str) -> list[dict[str, Any]] | None:
        try:
            raw = self._client.get(key)
        except Exception:
            logger.warning(
                "Redis GET failed endpoint=%s key=%s",
                self._endpoint,
                key,
                exc_info=True,
            )
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            logger.warning(
                "Redis cache value is not valid JSON endpoint=%s key=%s bytes=%s",
                self._endpoint,
                key,
                len(raw),
                exc_info=True,
            )
            return None

    def _set_json(self, key: str, value: list[dict[str, Any]], ttl_seconds: int) -> None:
        payload = json.dumps(value).encode("utf-8")
        try:
            self._client.set(key, payload, ex=ttl_seconds)
            logger.debug(
                "Redis SET endpoint=%s key=%s bytes=%s ttl=%ss",
                self._endpoint,
                key,
                len(payload),
                ttl_seconds,
            )
        except Exception:
            logger.warning(
                "Redis SET failed endpoint=%s key=%s bytes=%s ttl=%ss",
                self._endpoint,
                key,
                len(payload),
                ttl_seconds,
                exc_info=True,
            )

    def _get_polars(self, key: str) -> pl.DataFrame | None:
        try:
            raw = self._client.get(key)
        except Exception:
            logger.warning(
                "Redis GET failed endpoint=%s key=%s",
                self._endpoint,
                key,
                exc_info=True,
            )
            return None
        if raw is None:
            return None
        try:
            return _decode_polars(raw)
        except Exception:
            logger.warning(
                "Redis cache value is not valid Parquet endpoint=%s key=%s bytes=%s",
                self._endpoint,
                key,
                len(raw),
                exc_info=True,
            )
            return None

    def _set_polars(self, key: str, df: pl.DataFrame, ttl_seconds: int) -> None:
        payload = _encode_polars(df)
        try:
            self._client.set(key, payload, ex=ttl_seconds)
            logger.debug(
                "Redis SET endpoint=%s key=%s bytes=%s ttl=%ss",
                self._endpoint,
                key,
                len(payload),
                ttl_seconds,
            )
        except Exception:
            logger.warning(
                "Redis SET failed endpoint=%s key=%s bytes=%s ttl=%ss",
                self._endpoint,
                key,
                len(payload),
                ttl_seconds,
                exc_info=True,
            )


def get_cache_client() -> CacheClient | None:
    """Return a shared cache client, or None when Redis is unavailable or unset."""
    global _cache_client
    if _cache_client is not _UNSET:
        return _cache_client  # type: ignore[return-value]

    redis_url = os.getenv("REDIS_URL", "").strip()
    if not redis_url:
        logger.debug("Redis cache disabled: REDIS_URL is not set.")
        _cache_client = None
        return None

    endpoint = _redis_endpoint_label(redis_url)
    try:
        import redis

        logger.info("Connecting to Redis endpoint=%s", endpoint)
        client = redis.from_url(redis_url, decode_responses=False)
        client.ping()
        _cache_client = RedisCacheClient(client, endpoint=endpoint)
        logger.info("Redis cache enabled endpoint=%s", endpoint)
    except Exception:
        logger.warning(
            "Redis unavailable; caching disabled endpoint=%s",
            endpoint,
            exc_info=True,
        )
        _cache_client = None
    return _cache_client  # type: ignore[return-value]


def invalidate_user_games_cache(user_id: str) -> None:
    cache = get_cache_client()
    if cache is not None:
        logger.info("Invalidating Redis user games cache user_id=%s", user_id)
        cache.delete(user_games_cache_key(user_id))


def invalidate_user_accounts_cache(user_id: str) -> None:
    cache = get_cache_client()
    if cache is not None:
        logger.info("Invalidating Redis user accounts cache user_id=%s", user_id)
        cache.delete(user_accounts_cache_key(user_id))


def invalidate_user_games_and_accounts_cache(user_id: str) -> None:
    cache = get_cache_client()
    if cache is not None:
        logger.info("Invalidating Redis user games and accounts cache user_id=%s", user_id)
        cache.delete(user_games_cache_key(user_id), user_accounts_cache_key(user_id))


def invalidate_admin_log_aggregates_cache() -> None:
    cache = get_cache_client()
    if cache is not None:
        logger.info("Invalidating Redis admin log aggregates cache")
        cache.delete(
            log_level_hourly_counts_cache_key(),
            exception_hourly_counts_cache_key(),
        )


def reset_cache_client_for_tests() -> None:
    """Clear the module-level singleton (tests only)."""
    global _cache_client
    _cache_client = _UNSET
