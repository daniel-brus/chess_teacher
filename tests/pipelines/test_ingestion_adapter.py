from unittest.mock import MagicMock, patch

import pytest
import requests

from chess_teacher.pipelines.ingestion.adapter import (
    _USER_AGENT,
    ChessComAdapter,
    LichessAdapter,
)
from chess_teacher.platform.account import Account, AccountPlatform
from chess_teacher.utils.exception_utils import AdapterClientError, AdapterError


def _account(platform: AccountPlatform, username: str = "danielbrus") -> Account:
    return Account.from_username_and_platform(username, platform)


def test_lichess_adapter_sends_custom_user_agent() -> None:
    adapter = LichessAdapter(_account(AccountPlatform.LICHESS))
    headers = adapter._get_headers()
    assert headers["User-Agent"] == _USER_AGENT
    assert headers["Accept"] == "application/x-ndjson"


def test_chesscom_adapter_shares_user_agent_constant() -> None:
    adapter = ChessComAdapter(_account(AccountPlatform.CHESS_COM, username="ikbendaniel"))
    assert adapter._get_headers()["User-Agent"] == _USER_AGENT


@patch("chess_teacher.pipelines.ingestion.adapter.requests.get")
def test_adapter_raises_client_error_for_http_404(mock_get: MagicMock) -> None:
    response = MagicMock()
    response.status_code = 404
    response.raise_for_status.side_effect = requests.HTTPError(
        "404 Client Error",
        response=response,
    )
    mock_get.return_value = response

    adapter = LichessAdapter(_account(AccountPlatform.LICHESS))
    with pytest.raises(AdapterClientError, match="404"):
        adapter._get_response("https://lichess.org/api/games/user/danielbrus")


@patch("chess_teacher.pipelines.ingestion.adapter.requests.get")
def test_adapter_retries_allowed_for_http_429(mock_get: MagicMock) -> None:
    response = MagicMock()
    response.status_code = 429
    response.raise_for_status.side_effect = requests.HTTPError(
        "429 Too Many Requests",
        response=response,
    )
    mock_get.return_value = response

    adapter = LichessAdapter(_account(AccountPlatform.LICHESS))
    with pytest.raises(AdapterError) as exc_info:
        adapter._get_response("https://lichess.org/api/games/user/danielbrus")
    assert not isinstance(exc_info.value, AdapterClientError)
