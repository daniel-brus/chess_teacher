"""Tests for cache_utils."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import polars as pl
import pytest

from chess_teacher.platform.account import Account, AccountPlatform
from chess_teacher.utils import cache_utils
from chess_teacher.utils.cache_utils import (
    RedisCacheClient,
    _account_from_cache_dict,
    _account_to_cache_dict,
    _decode_polars,
    _encode_polars,
    _redis_endpoint_label,
    exception_hourly_counts_cache_key,
    get_cache_client,
    invalidate_admin_log_aggregates_cache,
    invalidate_user_games_and_accounts_cache,
    log_level_hourly_counts_cache_key,
    user_accounts_cache_key,
    user_games_cache_key,
)


@pytest.fixture(autouse=True)
def _reset_cache_singleton():
    cache_utils.reset_cache_client_for_tests()
    yield
    cache_utils.reset_cache_client_for_tests()


class TestCacheKeys:
    def test_user_games_cache_key(self):
        assert user_games_cache_key("abc") == "user:abc:games:v1"

    def test_user_accounts_cache_key(self):
        assert user_accounts_cache_key("abc") == "user:abc:accounts:v1"

    def test_admin_log_aggregate_cache_keys(self):
        assert log_level_hourly_counts_cache_key() == "admin:log_level_hourly_counts:v1"
        assert exception_hourly_counts_cache_key() == "admin:exception_hourly_counts:v1"

    def test_redis_endpoint_label_hides_credentials(self):
        label = _redis_endpoint_label("rediss://default:secret@fancy-marmot.upstash.io:6379")
        assert label == "fancy-marmot.upstash.io:6379"
        assert "secret" not in label


class TestAccountSerialization:
    def test_round_trip_with_latest_ingestion(self):
        account = Account(
            account_id="id1",
            username="hikaru",
            platform=AccountPlatform.CHESS_COM,
            latest_ingestion=datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC),
        )
        restored = _account_from_cache_dict(_account_to_cache_dict(account))
        assert restored == account

    def test_round_trip_without_latest_ingestion(self):
        account = Account.from_username_and_platform("hikaru", AccountPlatform.LICHESS)
        restored = _account_from_cache_dict(_account_to_cache_dict(account))
        assert restored == account


class TestPolarsSerialization:
    def test_round_trip(self):
        df = pl.DataFrame({"game_id": ["g1"], "result": ["win"]})
        restored = _decode_polars(_encode_polars(df))
        assert restored.equals(df)


class TestGetCacheClient:
    def test_returns_none_when_redis_url_missing(self, monkeypatch):
        monkeypatch.delenv("REDIS_URL", raising=False)
        assert get_cache_client() is None
        assert get_cache_client() is None

    def test_returns_none_when_redis_unavailable(self, monkeypatch):
        import sys

        monkeypatch.setenv("REDIS_URL", "redis://invalid-host:6379/0")
        fake_redis = MagicMock()
        fake_redis.from_url.side_effect = ConnectionError("down")
        monkeypatch.setitem(sys.modules, "redis", fake_redis)
        assert get_cache_client() is None


class TestRedisCacheClient:
    def test_set_and_get_user_accounts(self):
        redis_client = MagicMock()
        redis_client.get.return_value = None
        cache = RedisCacheClient(redis_client, endpoint="test-host:6379")

        account = Account.from_username_and_platform("hikaru", AccountPlatform.CHESS_COM)
        cache.set_user_accounts("user1", [account])

        set_args = redis_client.set.call_args
        assert set_args.args[0] == user_accounts_cache_key("user1")
        assert set_args.kwargs["ex"] == cache_utils.USER_ACCOUNTS_TTL_SECONDS

        stored = set_args.args[1].decode("utf-8")
        redis_client.get.return_value = stored.encode("utf-8")
        assert cache.get_user_accounts("user1") == [account]

    def test_set_and_get_user_games(self):
        redis_client = MagicMock()
        redis_client.get.return_value = None
        cache = RedisCacheClient(redis_client, endpoint="test-host:6379")

        games = pl.DataFrame({"game_id": ["g1"], "result": ["win"]})
        cache.set_user_games("user1", games)

        set_args = redis_client.set.call_args
        assert set_args.args[0] == user_games_cache_key("user1")
        assert set_args.kwargs["ex"] == cache_utils.USER_GAMES_TTL_SECONDS

        redis_client.get.return_value = set_args.args[1]
        restored = cache.get_user_games("user1")
        assert restored is not None
        assert restored.equals(games)

    def test_set_and_get_admin_log_aggregates(self):
        redis_client = MagicMock()
        redis_client.get.return_value = None
        cache = RedisCacheClient(redis_client, endpoint="test-host:6379")

        level_counts = pl.DataFrame({"bucket_start": [1], "log_count": [2]})
        exception_counts = pl.DataFrame({"bucket_start": [1], "exception_count": [3]})
        cache.set_log_level_hourly_counts(level_counts)
        cache.set_exception_hourly_counts(exception_counts)

        assert redis_client.set.call_args_list[0].args[0] == log_level_hourly_counts_cache_key()
        assert redis_client.set.call_args_list[1].args[0] == exception_hourly_counts_cache_key()

        redis_client.get.return_value = redis_client.set.call_args_list[0].args[1]
        assert cache.get_log_level_hourly_counts() is not None

    def test_delete(self):
        redis_client = MagicMock()
        cache = RedisCacheClient(redis_client, endpoint="test-host:6379")
        cache.delete("a", "b")
        redis_client.delete.assert_called_once_with("a", "b")


class TestInvalidation:
    def test_invalidate_user_games_and_accounts_cache(self, monkeypatch):
        cache = MagicMock()
        monkeypatch.setattr(cache_utils, "_cache_client", cache)
        invalidate_user_games_and_accounts_cache("user1")
        cache.delete.assert_called_once_with(
            user_games_cache_key("user1"),
            user_accounts_cache_key("user1"),
        )

    def test_invalidate_admin_log_aggregates_cache(self, monkeypatch):
        cache = MagicMock()
        monkeypatch.setattr(cache_utils, "_cache_client", cache)
        invalidate_admin_log_aggregates_cache()
        cache.delete.assert_called_once_with(
            log_level_hourly_counts_cache_key(),
            exception_hourly_counts_cache_key(),
        )
