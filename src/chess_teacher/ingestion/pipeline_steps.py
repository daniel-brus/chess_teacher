from datetime import datetime
from typing import Literal
from uuid import uuid4

from chess_teacher.ingestion.adapter import AdapterFactory
from chess_teacher.ingestion.raw_games import RawGame
from chess_teacher.ingestion.transformations import (
    ApplyChessComOpeningLookupTransformation,
    ApplyLichessOpeningNameTransformation,
    CleanPGNTransformation,
    DeriveOpeningTransformation,
    ExtractFileMetadataTransformation,
    ExtractGameMetadataTransformation,
    ExtractPlatformGameIdTransformation,
    ExtractPlayersAndResultTransformation,
    FilterGamesWithPGNTransformation,
)
from chess_teacher.other.dataclasses import RawEcoCode
from chess_teacher.pipelines.pipeline_base import PipelineContext, PipelineStep
from chess_teacher.pipelines.pipeline_steps import (
    LoadingStrategy,
    MergeStrategy,
    StorageToTableStep,
)
from chess_teacher.pipelines.transformations import (
    AssertUniqueColumnsTransformation,
    CreateHashedIdTransformation,
    JoinWithTableTransformation,
    RenameColumnsTransformation,
)
from chess_teacher.platform.account import Account
from chess_teacher.utils.db_client import DatabaseClient
from chess_teacher.utils.exception_utils import (
    AdapterError,
    DatabaseError,
    FileError,
    PipelineError,
)
from chess_teacher.utils.file_utils import FileType
from chess_teacher.utils.file_writer import FileWriter, FileWriterFactory
from chess_teacher.utils.general_utils import build_daily_key, get_current_datetime
from chess_teacher.utils.object_storage.base import ObjectStorage
from chess_teacher.utils.object_storage.factory import get_raw_storage

_INGESTION_FILE_TYPE = FileType.JSONL


def _get_account_storage_prefix(
    folder: Literal["ingested", "failed", "processed"], account: Account
) -> str:
    """Account-level storage prefix (all dates under YYYY/MM/DD subdirs)."""
    return ObjectStorage.resolve_key(folder, account.account_id)


def _get_daily_ingest_prefix(account: Account) -> str:
    """Today's ingest prefix where API stream writes new objects."""
    return build_daily_key(_get_account_storage_prefix("ingested", account))


def _fetch_account(db_client: DatabaseClient, context: PipelineContext) -> Account:
    if context.account_id is None:
        raise PipelineError("account_id is required for ingestion pipeline steps")
    return Account.fetch_from_db(db_client, id=context.account_id)


class IngestionFromAPIStreamStep(PipelineStep):
    """Ingest data from an API stream into object storage."""

    def __init__(self, *, storage: ObjectStorage | None = None) -> None:
        super().__init__(name="IngestionFromAPIStream")
        self._storage = storage

    def _get_storage(self) -> ObjectStorage:
        return self._storage if self._storage is not None else get_raw_storage()

    def _generate_filename(self, account: Account) -> str:
        """Generate the name of the output object."""
        return f"{account.platform.value}_{uuid4().hex}.{_INGESTION_FILE_TYPE.value}"

    def _get_last_updated(self, db_client: DatabaseClient, account: Account) -> datetime | None:
        """Fetch the last updated time from the database to get the most up-to-date value."""
        try:
            result = account.fetch_from_db(db_client, id=account.account_id).latest_ingestion
        except Exception as e:
            self.logger.log_and_raise(DatabaseError(f"Error fetching last updated time: {e}"))
        return result

    def _set_last_updated(
        self,
        db_client: DatabaseClient,
        account: Account,
        last_updated: datetime = get_current_datetime(),
    ) -> None:
        """Set the last updated time in the database."""
        try:
            account.upsert_latest(db_client, "latest_ingestion", last_updated)
        except Exception as e:
            self.logger.log_and_raise(DatabaseError(f"Error setting last updated time: {e}"))

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        account = _fetch_account(db_client, context)
        storage = self._get_storage()
        target_prefix = _get_daily_ingest_prefix(account)
        adapter = AdapterFactory.from_account(account)
        writer: FileWriter = FileWriterFactory.get_writer(_INGESTION_FILE_TYPE, logger=self.logger)

        output_key = ObjectStorage.resolve_key(target_prefix, self._generate_filename(account))
        since = self._get_last_updated(db_client, account)
        since_new = get_current_datetime()

        context.progress_update(f"Fetching new games from {account.platform.value}...")
        try:
            records = adapter.get_records(since=since)
            if not records:
                self.logger.info(f"[{self.name}] No records to write.")
                context.progress_update("No new games from the platform.")
                return
        except Exception as e:
            self.logger.log_and_raise(AdapterError(f"Error getting records: {e}"))

        context.progress_update(
            f"Writing {len(records)} game{'s' if len(records) != 1 else ''} to storage..."
        )
        writer.write(records, output_key, storage)
        self.logger.info(f"[{self.name}] Written to {output_key}.")
        self._set_last_updated(db_client, account, since_new)
        self.logger.info(f"[{self.name}] Ingestion completed.")
        context.progress_pop()
        context.progress_success(f"Saved {len(records):,} game(s) to storage.")


class LoadIngestedFilesToDB(StorageToTableStep):
    """Load ingested files to the database."""

    def __init__(self, *, storage: ObjectStorage | None = None) -> None:
        super().__init__(
            name="LoadIngestedFilesToDB",
            storage_path="",
            file_type=_INGESTION_FILE_TYPE,
            data_class=RawGame,
            storage=storage,
            transformations=[
                FilterGamesWithPGNTransformation(),
                RenameColumnsTransformation({"pgn": "raw_pgn"}),
                ExtractFileMetadataTransformation(),
                JoinWithTableTransformation(with_data_class=Account),
                ExtractPlatformGameIdTransformation(),
                CreateHashedIdTransformation(data_class=RawGame),
                ExtractGameMetadataTransformation(),
                ApplyLichessOpeningNameTransformation(),
                ExtractPlayersAndResultTransformation(),
                CleanPGNTransformation(),
                JoinWithTableTransformation(
                    with_data_class=RawEcoCode,
                    left_on=["eco_code"],
                    right_on=["eco_code"],
                ),
                DeriveOpeningTransformation(),
                ApplyChessComOpeningLookupTransformation(),
                AssertUniqueColumnsTransformation("game_id", label="game_id"),
            ],
            loading_strategy=LoadingStrategy.MERGE,
            merge_strategy=MergeStrategy.upsert(),
        )

    def _resolve_storage_paths(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        account = _fetch_account(db_client, context)
        self.storage_path = _get_account_storage_prefix("ingested", account)
        self.quarantine_path = _get_account_storage_prefix("failed", account)


class ArchiveIngestedFilesStep(PipelineStep):
    """
    Move successfully processed ingested objects from ``ingested/`` to ``processed/``.

    Lists every ``.jsonl`` object under the account's ingested prefix (not only keys
    recorded during load) so nothing is left behind after a successful pipeline run.
    Uses :meth:`ObjectStorage.move_verified` so the source key is gone after each move.
    """

    file_type: FileType = FileType.JSONL

    def __init__(
        self,
        *,
        recursive: bool = True,
        glob_pattern: str | None = None,
        storage: ObjectStorage | None = None,
    ) -> None:
        super().__init__(name="ArchiveIngestedFiles")
        self.recursive = recursive
        self.glob_pattern = glob_pattern
        self._storage = storage

    def _get_storage(self) -> ObjectStorage:
        return self._storage if self._storage is not None else get_raw_storage()

    def _archive_destination(self, source_key: str, source_prefix: str, archive_prefix: str) -> str:
        """Preserve relative layout under source_prefix in the archive."""
        relative = ObjectStorage.relative_key_under(source_key, source_prefix)
        destination = ObjectStorage.resolve_key(archive_prefix, relative)
        storage = self._get_storage()
        if storage.read_bytes(destination) is not None:
            destination = ObjectStorage.resolve_key(
                archive_prefix,
                ObjectStorage.unique_key_variant(relative, uuid4().hex),
            )
        return destination

    def _list_ingested_keys(self, storage: ObjectStorage, source_prefix: str) -> list[str]:
        return storage.list_keys(
            source_prefix,
            recursive=self.recursive,
            suffix=self.file_type.value,
            glob_pattern=self.glob_pattern,
        )

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        account = _fetch_account(db_client, context)
        storage = self._get_storage()
        source_prefix = _get_account_storage_prefix("ingested", account)
        archive_prefix = _get_account_storage_prefix("processed", account)

        keys = self._list_ingested_keys(storage, source_prefix)
        loaded_count = len(context.loaded_storage_keys)
        if loaded_count and loaded_count != len(keys):
            self.logger.warning(
                f"[{self.name}] Loaded {loaded_count} file(s) but found {len(keys)} "
                f"under {source_prefix}; archiving all listed objects."
            )

        if not keys:
            self.logger.info(f"[{self.name}] No objects to archive under {source_prefix}.")
            context.progress_pop()
            context.progress_warning("No ingested files to archive.")
            return

        file_total = len(keys)
        archived = 0

        for file_index, key in enumerate(keys, start=1):
            label = ObjectStorage.key_basename(key)
            context.progress_update(f"Archiving file {file_index}/{file_total}: {label}...")
            destination = self._archive_destination(key, source_prefix, archive_prefix)
            try:
                storage.move_verified(key, destination, overwrite=False)
            except FileError as e:
                self.logger.log_and_raise(
                    FileError(f"Failed to archive {key} to {destination}: {e}")
                )
            self.logger.info(f"[{self.name}] Archived {key} -> {destination}.")
            archived += 1

        leftover = self._list_ingested_keys(storage, source_prefix)
        if leftover:
            self.logger.warning(
                f"[{self.name}] {len(leftover)} object(s) still under {source_prefix} "
                f"after archive; forcing delete."
            )
            storage.delete_keys(leftover, missing_ok=False)

        self.logger.info(f"[{self.name}] Archived {archived} object(s) to {archive_prefix}.")
        context.progress_pop()
        context.progress_success(f"Archived {archived} file(s).")
