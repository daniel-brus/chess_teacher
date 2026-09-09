import json
import logging

import polars as pl
import pytest

from chess_teacher.pipelines.preprocessing.transformations import (
    ExpandRawResponseTransformation,
    _stringify_mixed_type_fields,
)
from chess_teacher.utils.exception_utils import TransformationError

_FEN = "nrnbkqbr/pppppppp/8/8/8/8/PPPPPPPP/NRNBKQBR w KQkq - 0 1"


def _raw_df(payloads: list[dict]) -> pl.DataFrame:
    n = len(payloads)
    return pl.DataFrame({
        "game_id": [f"g{i}" for i in range(n)],
        "platform_game_id": [f"p{i}" for i in range(n)],
        "account_id": ["acct"] * n,
        "raw_response": [json.dumps(p) for p in payloads],
        "source_file": ["ingested/x.jsonl"] * n,
        "ingested_at": ["2024-01-01T12:00:00+00:00"] * n,
    })


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
    """Full-scan infer: int then FEN string past Polars' default 100-row window."""
    payloads = [{"id": i, "initialFen": i} for i in range(100)]
    payloads.append({"id": 100, "initialFen": _FEN})

    result = ExpandRawResponseTransformation().transform(_raw_df(payloads))

    assert result.height == 101
    assert result["initialFen"][-1] == _FEN


def test_expand_raw_response_stringifies_struct_vs_scalar_clash(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Struct vs str cannot unify even with full-scan infer → stringify fallback."""
    payloads = [
        {"id": 1, "variant": {"key": "standard"}},
        {"id": 2, "variant": _FEN},
    ]

    with caplog.at_level(logging.WARNING):
        result = ExpandRawResponseTransformation().transform(_raw_df(payloads))

    assert result.height == 2
    assert isinstance(result["variant"][0], str)
    assert json.loads(result["variant"][0]) == {"key": "standard"}
    assert result["variant"][1] == _FEN
    assert any(
        "stringifying mixed columns" in record.message and "variant" in record.message
        for record in caplog.records
    )


def test_expand_raw_response_keeps_nulls_when_other_rows_typed() -> None:
    payloads = [
        {"id": 1, "initialFen": None},
        {"id": 2},  # key absent
        {"id": 3, "initialFen": _FEN},
    ]
    result = ExpandRawResponseTransformation().transform(_raw_df(payloads))
    assert result.height == 3
    assert result["initialFen"][0] is None
    assert result["initialFen"][2] == _FEN


def test_expand_raw_response_bool_and_int_are_mixed_types() -> None:
    """type(True) is bool, type(1) is int — must stringify, not treat as homogeneous."""
    rows = [{"flag": True}, {"flag": 1}]
    normalized, mixed = _stringify_mixed_type_fields(rows)
    assert mixed == frozenset({"flag"})
    assert normalized[0]["flag"] == "True"
    assert normalized[1]["flag"] == "1"


def test_expand_raw_response_list_vs_dict_json_dumps_both() -> None:
    rows = [{"x": [1, 2]}, {"x": {"a": 1}}]
    normalized, mixed = _stringify_mixed_type_fields(rows)
    assert mixed == frozenset({"x"})
    assert json.loads(normalized[0]["x"]) == [1, 2]
    assert json.loads(normalized[1]["x"]) == {"a": 1}


def test_stringify_mixed_type_fields_noop_when_homogeneous() -> None:
    rows = [{"a": 1}, {"a": 2}]
    out, mixed = _stringify_mixed_type_fields(rows)
    assert out is rows
    assert mixed == frozenset()


def test_expand_raw_response_empty_frame() -> None:
    df = pl.DataFrame({
        "game_id": pl.Series([], dtype=pl.Utf8),
        "raw_response": pl.Series([], dtype=pl.Utf8),
    })
    result = ExpandRawResponseTransformation().transform(df)
    assert result.height == 0
    assert result.columns == df.columns


def test_expand_raw_response_rejects_non_object_json() -> None:
    df = pl.DataFrame({
        "game_id": ["g1"],
        "raw_response": [json.dumps([1, 2, 3])],
    })
    with pytest.raises(TransformationError, match="object"):
        ExpandRawResponseTransformation().transform(df)


def test_expand_raw_response_rejects_invalid_json() -> None:
    df = pl.DataFrame({"game_id": ["g1"], "raw_response": ["not-json{"]})
    with pytest.raises(TransformationError, match="parse"):
        ExpandRawResponseTransformation().transform(df)


def test_expand_raw_response_requires_column() -> None:
    df = pl.DataFrame({"game_id": ["g1"]})
    with pytest.raises(TransformationError, match="raw_response"):
        ExpandRawResponseTransformation().transform(df)
