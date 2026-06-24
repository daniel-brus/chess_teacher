from unittest.mock import MagicMock

import polars as pl

from chess_teacher.pipelines.preprocessing.games import Game
from chess_teacher.pipelines.preprocessing.moves import Move
from chess_teacher.utils.pipeline_utils.transformations import IncrementalFilterTransformation


def test_incremental_filter_skips_existing_game_ids_in_games() -> None:
    df = pl.DataFrame({
        "game_id": ["game-1", "game-2", "game-3"],
        "account_id": ["acct-1", "acct-1", "acct-1"],
        "raw_response": ["{}", "{}", "{}"],
    })
    db_client = MagicMock()
    db_client.table_exists.return_value = True
    db_client.engine.execute_parameterized_query.return_value = [
        {"game_id": "game-1"},
    ]

    filter = IncrementalFilterTransformation(
        target_data_class=Game,
        on="game_id",
        db_client=db_client,
    )
    filter.set_scope_where("\"account_id\" = 'acct-1'")
    result = filter.transform(df)

    assert result.height == 2
    assert result["game_id"].to_list() == ["game-2", "game-3"]
    sql = db_client.engine.execute_parameterized_query.call_args[0][0]
    assert "\"account_id\" = 'acct-1'" in sql


def test_incremental_filter_keeps_all_when_target_table_missing() -> None:
    df = pl.DataFrame({
        "game_id": ["game-1"],
        "account_id": ["acct-1"],
        "raw_response": ["{}"],
    })
    db_client = MagicMock()
    db_client.table_exists.return_value = False

    result = IncrementalFilterTransformation(
        target_data_class=Game,
        on="game_id",
        db_client=db_client,
    ).transform(df)

    assert result.height == 1
    db_client.engine.execute_parameterized_query.assert_not_called()


def test_incremental_filter_skips_existing_game_ids_in_moves() -> None:
    df = pl.DataFrame({
        "game_id": ["game-1", "game-2", "game-3"],
        "account_id": ["acct-1", "acct-1", "acct-1"],
    })
    db_client = MagicMock()
    db_client.table_exists.return_value = True
    db_client.engine.execute_parameterized_query.return_value = [
        {"game_id": "game-1"},
        {"game_id": "game-2"},
    ]

    filter = IncrementalFilterTransformation(
        target_data_class=Move,
        on="game_id",
        db_client=db_client,
    )
    filter.set_scope_where("\"account_id\" = 'acct-1'")
    result = filter.transform(df)

    assert result.height == 1
    assert result["game_id"].to_list() == ["game-3"]


def test_incremental_filter_noop_when_on_not_configured() -> None:
    df = pl.DataFrame({
        "game_id": ["game-1", "game-2"],
        "account_id": ["acct-1", "acct-1"],
    })
    db_client = MagicMock()

    result = IncrementalFilterTransformation(
        target_data_class=Game,
        on=None,
        db_client=db_client,
    ).transform(df)

    assert result.height == 2
    db_client.table_exists.assert_not_called()


def test_incremental_filter_supports_different_source_column() -> None:
    df = pl.DataFrame({
        "source_game_id": ["game-1", "game-2"],
        "account_id": ["acct-1", "acct-1"],
    })
    db_client = MagicMock()
    db_client.table_exists.return_value = True
    db_client.engine.execute_parameterized_query.return_value = [
        {"game_id": "game-1"},
    ]

    filter = IncrementalFilterTransformation(
        target_data_class=Game,
        on="game_id",
        source_column="source_game_id",
        db_client=db_client,
    )
    filter.set_scope_where("\"account_id\" = 'acct-1'")
    result = filter.transform(df)

    assert result.height == 1
    assert result["source_game_id"].to_list() == ["game-2"]
