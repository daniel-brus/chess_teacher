"""Ingestion pipeline steps x PipelineMode (mocked storage / DB)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import polars as pl
import pytest

from chess_teacher.pipelines.ingestion.pipeline_steps import LoadIngestedFilesToDB
from chess_teacher.pipelines.modes import (
    PIPELINE_MODES,
    PipelineMode,
    StorageFolder,
    ingestion_load_merge_strategy,
    ingestion_load_source_folders,
)
from chess_teacher.platform.account import Account, AccountPlatform
from chess_teacher.utils.db.client import MergeStrategy, WriteResult, WriteStrategy
from chess_teacher.utils.metadata_utils import TableMetadata
from chess_teacher.utils.pipeline_utils.pipeline_base import PipelineContext
from chess_teacher.utils.pipeline_utils.transformations import JoinWithTableTransformation

_ACCOUNT_ID = "acct-1"
_USER = "TestPlayer"
_INGESTION_TS = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)


def _account() -> Account:
    return Account(
        account_id=_ACCOUNT_ID,
        username=_USER,
        platform=AccountPlatform.LICHESS,
    )


def _key(folder: StorageFolder, name: str = "Lichess_batch.jsonl") -> str:
    return f"{folder}/{_ACCOUNT_ID}/2024/06/01/{name}"


@pytest.mark.parametrize("mode", PIPELINE_MODES)
def test_load_ingested_merge_strategy_matches_mode(mode: PipelineMode) -> None:
    step = LoadIngestedFilesToDB(mode=mode, storage=MagicMock())
    assert step.merge_strategy == ingestion_load_merge_strategy(mode)
    assert step.mode == mode


@pytest.mark.parametrize(
    ("mode", "expected_folders"),
    [
        (PipelineMode.INCREMENTAL, ("ingested",)),
        (PipelineMode.RETRY, ("ingested", "failed")),
        (PipelineMode.REPROCESS, ("ingested", "failed", "processed")),
        (PipelineMode.FULL_RELOAD, ("ingested", "failed", "processed")),
    ],
)
def test_list_source_keys_scans_mode_folders(
    monkeypatch: pytest.MonkeyPatch,
    mode: PipelineMode,
    expected_folders: tuple[StorageFolder, ...],
) -> None:
    storage = MagicMock()
    listed_prefixes: list[str] = []

    def fake_list_keys(
        prefix: str,
        *,
        recursive: bool = True,
        suffix: str | None = None,
        glob_pattern: str | None = None,
    ) -> list[str]:
        del recursive, suffix, glob_pattern
        listed_prefixes.append(prefix)
        # One key per folder so we can assert membership later.
        folder = prefix.split("/", 1)[0]
        assert folder in ("ingested", "failed", "processed")
        return [_key(folder)]  # type: ignore[arg-type]

    storage.list_keys.side_effect = fake_list_keys

    step = LoadIngestedFilesToDB(mode=mode, storage=storage)
    step._account = _account()
    step._ingested_prefix = f"ingested/{_ACCOUNT_ID}"
    step._failed_prefix = f"failed/{_ACCOUNT_ID}"
    step._processed_prefix = f"processed/{_ACCOUNT_ID}"

    keys = step._list_source_keys()

    assert tuple(p.split("/", 1)[0] for p in listed_prefixes) == expected_folders
    assert keys == [_key(folder) for folder in expected_folders]
    assert ingestion_load_source_folders(mode) == expected_folders


def test_full_reload_sets_match_condition(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        Account,
        "fetch_from_db",
        classmethod(lambda cls, db, id: _account()),
    )
    step = LoadIngestedFilesToDB(mode=PipelineMode.FULL_RELOAD, storage=MagicMock())
    step._resolve_storage_paths(MagicMock(), PipelineContext(account_id=_ACCOUNT_ID))
    assert step.match_condition is not None
    assert _ACCOUNT_ID in step.match_condition
    assert step.merge_strategy == MergeStrategy.full_sync()


def test_incremental_does_not_set_match_condition(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        Account,
        "fetch_from_db",
        classmethod(lambda cls, db, id: _account()),
    )
    step = LoadIngestedFilesToDB(mode=PipelineMode.INCREMENTAL, storage=MagicMock())
    step._resolve_storage_paths(MagicMock(), PipelineContext(account_id=_ACCOUNT_ID))
    assert step.match_condition is None
    assert step.merge_strategy == MergeStrategy.upsert()


def test_list_source_keys_dedupes_identical_key_across_folders() -> None:
    """Same object key listed under two prefixes must appear once."""
    storage = MagicMock()
    shared = _key("ingested")

    def fake_list_keys(
        prefix: str,
        *,
        recursive: bool = True,
        suffix: str | None = None,
        glob_pattern: str | None = None,
    ) -> list[str]:
        del recursive, suffix, glob_pattern, prefix
        return [shared]

    storage.list_keys.side_effect = fake_list_keys
    step = LoadIngestedFilesToDB(mode=PipelineMode.RETRY, storage=storage)
    step._account = _account()
    step._ingested_prefix = f"ingested/{_ACCOUNT_ID}"
    step._failed_prefix = f"failed/{_ACCOUNT_ID}"
    step._processed_prefix = f"processed/{_ACCOUNT_ID}"

    keys = step._list_source_keys()
    assert keys == [shared]


def test_list_source_keys_empty_folders() -> None:
    storage = MagicMock()
    storage.list_keys.return_value = []
    step = LoadIngestedFilesToDB(mode=PipelineMode.FULL_RELOAD, storage=storage)
    step._account = _account()
    step._ingested_prefix = f"ingested/{_ACCOUNT_ID}"
    step._failed_prefix = f"failed/{_ACCOUNT_ID}"
    step._processed_prefix = f"processed/{_ACCOUNT_ID}"
    assert step._list_source_keys() == []


def test_load_ingested_empty_source_skips_save(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        Account,
        "fetch_from_db",
        classmethod(lambda cls, db, id: _account()),
    )
    storage = MagicMock()
    storage.list_keys.return_value = []
    step = LoadIngestedFilesToDB(mode=PipelineMode.INCREMENTAL, storage=storage)

    saved: list[pl.DataFrame] = []

    def capture_save(
        _db: MagicMock,
        _meta: object,
        data: pl.DataFrame,
    ) -> WriteResult:
        saved.append(data)
        return WriteResult(strategy=WriteStrategy.MERGE, rows_inserted=data.height)

    monkeypatch.setattr(step, "_load_records", lambda _db, _ctx: pl.DataFrame())
    monkeypatch.setattr(step, "_save_records", capture_save)
    monkeypatch.setattr(step, "_move_keys", lambda *_a, **_k: None)
    monkeypatch.setattr(step, "_ensure_ingested_empty", lambda: None)

    step.run(MagicMock(), PipelineContext(account_id=_ACCOUNT_ID))
    assert saved == []


def test_load_ingested_transform_save_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Full step.run with canned ingested rows; Account join mocked."""
    monkeypatch.setattr(
        "chess_teacher.utils.pipeline_utils.transformations.get_db_client",
        lambda: MagicMock(),
    )
    monkeypatch.setattr(
        Account,
        "fetch_from_db",
        classmethod(lambda cls, db, id: _account()),
    )

    storage = MagicMock()
    storage.list_keys.return_value = []
    step = LoadIngestedFilesToDB(mode=PipelineMode.INCREMENTAL, storage=storage)

    source_key = _key("ingested")
    source_df = pl.DataFrame({
        "id": ["lich-1", "lich-2"],
        "status": ["resign", "mate"],
        "_source_file": [source_key, source_key],
        "_ingestion_ts": [_INGESTION_TS, _INGESTION_TS],
    })

    db = MagicMock()
    db.ensure_metadata.return_value = None
    db.table_exists.return_value = True

    def fake_read(
        table: TableMetadata,
        columns: list[str] | None = None,
        where: str | None = None,
        as_polars: bool = True,
    ) -> pl.DataFrame:
        del where, as_polars
        assert table.qualified_name_sql() == Account.get_metadata().qualified_name_sql()
        frame = pl.DataFrame({
            "account_id": [_ACCOUNT_ID],
            "username": [_USER],
            "platform": [AccountPlatform.LICHESS.value],
        })
        if columns is not None:
            return frame.select([c for c in columns if c in frame.columns])
        return frame

    db.read.side_effect = fake_read

    for transformation in step.transformations:
        if isinstance(transformation, JoinWithTableTransformation):
            transformation.db_client = db

    saved: list[pl.DataFrame] = []

    def capture_save(
        _db: MagicMock,
        _meta: object,
        data: pl.DataFrame,
    ) -> WriteResult:
        saved.append(data)
        return WriteResult(strategy=WriteStrategy.MERGE, rows_inserted=data.height)

    monkeypatch.setattr(step, "_load_records", lambda _db, _ctx: source_df.clone())
    monkeypatch.setattr(step, "_save_records", capture_save)
    monkeypatch.setattr(step, "_move_keys", lambda *_a, **_k: None)
    monkeypatch.setattr(step, "_ensure_ingested_empty", lambda: None)

    step.run(db, PipelineContext(account_id=_ACCOUNT_ID))

    assert len(saved) == 1
    out = saved[0]
    assert out.height == 2
    assert out["account_id"].to_list() == [_ACCOUNT_ID, _ACCOUNT_ID]
    assert out["platform_game_id"].to_list() == ["lich-1", "lich-2"]
    assert out["raw_response"].null_count() == 0
    assert "game_id" in out.columns
    assert step.merge_strategy == MergeStrategy.upsert()
