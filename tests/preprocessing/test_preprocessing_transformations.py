import json

import polars as pl
import pytest

from chess_teacher.pipelines.preprocessing.transformations import (
    ExpandRawResponseTransformation,
    _stringify_mixed_type_fields,
)
from chess_teacher.utils.exception_utils import TransformationError


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


def test_expand_raw_response_handles_type_change_past_default_infer_window() -> None:
    """Prod: Polars default infer_schema_length=100; late FEN string broke expand."""
    fen = "nrnbkqbr/pppppppp/8/8/8/8/PPPPPPPP/NRNBKQBR w KQkq - 0 1"
    payloads = [{"id": i, "initialFen": i} for i in range(100)]
    payloads.append({"id": 100, "initialFen": fen})
    df = pl.DataFrame({
        "game_id": [f"g{i}" for i in range(101)],
        "platform_game_id": [f"p{i}" for i in range(101)],
        "account_id": ["acct"] * 101,
        "raw_response": [json.dumps(p) for p in payloads],
        "source_file": ["ingested/x.jsonl"] * 101,
        "ingested_at": ["2024-01-01T12:00:00+00:00"] * 101,
    })

    result = ExpandRawResponseTransformation().transform(df)

    assert result.height == 101
    assert result["initialFen"][-1] == fen


def test_expand_raw_response_stringifies_struct_vs_scalar_clash() -> None:
    fen = "nrnbkqbr/pppppppp/8/8/8/8/PPPPPPPP/NRNBKQBR w KQkq - 0 1"
    payloads = [
        {"id": 1, "variant": {"key": "standard"}},
        {"id": 2, "variant": fen},
    ]
    df = pl.DataFrame({
        "game_id": ["g1", "g2"],
        "platform_game_id": ["p1", "p2"],
        "account_id": ["acct", "acct"],
        "raw_response": [json.dumps(p) for p in payloads],
        "source_file": ["ingested/x.jsonl", "ingested/x.jsonl"],
        "ingested_at": ["2024-01-01T12:00:00+00:00"] * 2,
    })

    result = ExpandRawResponseTransformation().transform(df)

    assert result.height == 2
    assert isinstance(result["variant"][0], str)
    assert fen in result["variant"][1]


def test_stringify_mixed_type_fields_noop_when_homogeneous() -> None:
    rows = [{"a": 1}, {"a": 2}]
    assert _stringify_mixed_type_fields(rows) is rows


def test_expand_raw_response_rejects_non_object_json() -> None:
    df = pl.DataFrame({
        "game_id": ["g1"],
        "raw_response": [json.dumps([1, 2, 3])],
    })
    with pytest.raises(TransformationError, match="object"):
        ExpandRawResponseTransformation().transform(df)
