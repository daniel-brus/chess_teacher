from __future__ import annotations

import pytest

from chess_teacher.platform.account import (
    Account,
    AccountPlatform,
    normalize_platform_username,
)


def test_normalize_lowercases_and_strips_at_prefix() -> None:
    assert normalize_platform_username("  @Hikaru  ", AccountPlatform.CHESS_COM) == "hikaru"
    account = Account.from_username_and_platform("DanielBrus", AccountPlatform.LICHESS)
    assert account.username == "danielbrus"


@pytest.mark.parametrize(
    ("username", "platform"),
    [
        ("ab", AccountPlatform.CHESS_COM),
        ("this-username-is-way-too-long-for-chesscom", AccountPlatform.CHESS_COM),
        ("bad name", AccountPlatform.LICHESS),
        ("https://lichess.org/danielbrus", AccountPlatform.LICHESS),
        ("", AccountPlatform.CHESS_COM),
        ("user/name", AccountPlatform.CHESS_COM),
    ],
)
def test_normalize_rejects_invalid_usernames(username: str, platform: AccountPlatform) -> None:
    with pytest.raises(ValueError):
        normalize_platform_username(username, platform)


def test_normalize_accepts_typical_usernames() -> None:
    assert normalize_platform_username("ikbendaniel", AccountPlatform.CHESS_COM) == "ikbendaniel"
    assert normalize_platform_username("a", AccountPlatform.LICHESS) == "a"
    assert normalize_platform_username("player_one", AccountPlatform.LICHESS) == "player_one"
