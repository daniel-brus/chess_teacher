"""Shared pipeline run modes and per-pipeline configuration helpers."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from chess_teacher.utils.db.client import MergeStrategy

StorageFolder = Literal["ingested", "failed", "processed"]


class PipelineMode(StrEnum):
    INCREMENTAL = "incremental"
    RETRY = "retry"
    REPROCESS = "reprocess"
    FULL_RELOAD = "full_reload"


PIPELINE_MODES = tuple(PipelineMode)


def preprocessing_transform_config(
    mode: PipelineMode,
    *,
    incremental_on: str,
) -> tuple[str | None, MergeStrategy]:
    """
    Map a pipeline run mode to incremental-filter and target-write settings.

    - incremental / retry: upsert + skip rows already in target (``on=incremental_on``)
    - reprocess: upsert + no incremental filter (``on=None``)
    - full_reload: full_sync + no incremental filter (``on=None``)
    """
    if mode in (PipelineMode.INCREMENTAL, PipelineMode.RETRY):
        return incremental_on, MergeStrategy.upsert()
    if mode == PipelineMode.REPROCESS:
        return None, MergeStrategy.upsert()
    return None, MergeStrategy.full_sync()


def ingestion_load_source_folders(mode: PipelineMode) -> tuple[StorageFolder, ...]:
    """Return storage folders to scan when loading ingested JSONL into the database."""
    if mode == PipelineMode.INCREMENTAL:
        return ("ingested",)
    if mode == PipelineMode.RETRY:
        return ("ingested", "failed")
    return ("ingested", "failed", "processed")


def ingestion_load_merge_strategy(mode: PipelineMode) -> MergeStrategy:
    """Return the merge strategy for loading raw games from storage."""
    if mode == PipelineMode.FULL_RELOAD:
        return MergeStrategy.full_sync()
    return MergeStrategy.upsert()
