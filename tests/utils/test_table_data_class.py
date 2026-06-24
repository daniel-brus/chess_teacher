"""Ensure TableDataClass dataclass fields stay in sync with metadata.yml."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from chess_teacher.pipelines.ingestion.raw_games import RawGame
from chess_teacher.pipelines.preprocessing.games import Game
from chess_teacher.pipelines.preprocessing.moves import Move
from chess_teacher.platform.account import Account
from chess_teacher.platform.user import User
from chess_teacher.platform.user_account import UserAccount
from chess_teacher.utils.pipeline_utils.pipeline_helpers import (
    PipelineRunResult,
    PipelineRunStepResult,
)
from chess_teacher.utils.table_data_class import TableDataClass

CLASSES_TO_TEST: list[type[TableDataClass]] = [
    User,
    Account,
    UserAccount,
    PipelineRunStepResult,
    PipelineRunResult,
    RawGame,
    Game,
    Move,
]


@pytest.mark.parametrize("model_cls", CLASSES_TO_TEST)
class TestMetadataSync:
    """Dataclass fields vs metadata.yml for each TableDataClass."""

    def test_metadata_in_sync(self, model_cls: type[TableDataClass]) -> None:
        errors = model_cls.validate_metadata_sync()
        assert not errors, "\n  ".join(errors)


def test_fetch_all_from_db_maps_rows_and_ensures_table() -> None:
    db_client = MagicMock()
    db_client.read.return_value = [
        {
            "account_id": "acct-1",
            "username": "player",
            "platform": "Chess.com",
            "latest_ingestion": None,
        }
    ]

    accounts = Account.fetch_all_from_db(db_client, order_by="account_id")

    db_client.ensure_table.assert_called_once_with(Account.get_metadata())
    db_client.read.assert_called_once_with(
        Account.get_metadata(),
        where=None,
        order_by="account_id",
        limit=None,
    )
    assert len(accounts) == 1
    assert accounts[0].account_id == "acct-1"
    assert accounts[0].username == "player"
