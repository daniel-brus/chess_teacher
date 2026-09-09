"""COPY merge must serialize jsonb dicts — VALUES path already did; COPY did not."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from chess_teacher.utils.db.client import (
    MERGE_COPY_THRESHOLD,
    DatabaseClient,
    MergeStrategy,
    _prepare_records_for_copy,
)
from chess_teacher.utils.metadata_utils import ColumnMetadata, TableMetadata


def _jsonb_table() -> TableMetadata:
    return TableMetadata(
        schema_name="games",
        table_name="move_characteristics",
        columns=(
            ColumnMetadata(name="move_id", data_type="text", nullable=False),
            ColumnMetadata(name="candidate_evaluations", data_type="jsonb", nullable=True),
        ),
        primary_key=("move_id",),
    )


def test_prepare_records_for_copy_json_dumps_dicts() -> None:
    payload = {"depth": 1, "evals_white_pov": {"e2e4": 0.2}}
    records = [{"move_id": "m1", "candidate_evaluations": payload}]
    prepared = _prepare_records_for_copy(records, _jsonb_table())
    assert prepared[0]["move_id"] == "m1"
    assert prepared[0]["candidate_evaluations"] == json.dumps(payload, ensure_ascii=False)
    assert isinstance(prepared[0]["candidate_evaluations"], str)


def test_prepare_records_for_copy_leaves_non_json_alone() -> None:
    table = TableMetadata(
        schema_name="s",
        table_name="t",
        columns=(ColumnMetadata(name="move_id", data_type="text", nullable=False),),
        primary_key=("move_id",),
    )
    records = [{"move_id": "m1"}]
    assert _prepare_records_for_copy(records, table) is records


def test_merge_via_copy_serializes_jsonb_before_copy() -> None:
    """Regression: prod failed with cannot adapt type 'dict' on COPY >1000 rows."""
    engine = MagicMock()
    conn = MagicMock()
    begin_cm = MagicMock()
    begin_cm.__enter__.return_value = conn
    begin_cm.__exit__.return_value = False
    engine.begin.return_value = begin_cm
    conn.execute.return_value.mappings.return_value.first.return_value = {
        "matched_count": 0,
        "delete_count": 0,
    }

    client = DatabaseClient(engine=engine)
    payload = {"depth": 8, "evals_white_pov": {"e2e4": 0.1}}
    records = [
        {"move_id": f"m{i}", "candidate_evaluations": dict(payload)}
        for i in range(MERGE_COPY_THRESHOLD + 1)
    ]

    client.merge(records, _jsonb_table(), strategy=MergeStrategy())

    copy_call = engine.copy_records.call_args
    assert copy_call is not None
    copied_records = copy_call.args[3]
    assert len(copied_records) == MERGE_COPY_THRESHOLD + 1
    assert isinstance(copied_records[0]["candidate_evaluations"], str)
    assert json.loads(copied_records[0]["candidate_evaluations"]) == payload
