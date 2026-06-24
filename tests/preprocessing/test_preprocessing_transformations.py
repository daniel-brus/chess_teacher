import json

import polars as pl

from chess_teacher.pipelines.preprocessing.transformations import ExpandRawResponseTransformation


def test_expand_raw_response_parses_json_and_preserves_identity_columns() -> None:
    payload = {"pgn": "1. e4 e5", "uuid": "abc-123"}
    df = pl.DataFrame({
        "game_id": ["game-1"],
        "platform_game_id": ["abc-123"],
        "account_id": ["acct-1"],
        "raw_response": [json.dumps(payload)],
        "source_file": ["ingested/acct-1/2024/01/01/chess_com_x.jsonl"],
        "ingested_at": ["2024-01-01T12:00:00+00:00"],
    })

    result = ExpandRawResponseTransformation().transform(df)

    assert result.height == 1
    row = result.row(0, named=True)
    assert row["pgn"] == "1. e4 e5"
    assert row["uuid"] == "abc-123"
    assert row["game_id"] == "game-1"
    assert row["account_id"] == "acct-1"
    assert row["platform_game_id"] == "abc-123"
