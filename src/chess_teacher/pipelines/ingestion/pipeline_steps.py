from datetime import datetime
from typing import Literal
from uuid import uuid4

import polars as pl

from chess_teacher.pipelines.ingestion.adapter import AdapterFactory
from chess_teacher.pipelines.ingestion.raw_games import RawGame
from chess_teacher.pipelines.ingestion.transformations import (
    ExtractFileMetadataTransformation,
    ExtractPlatformGameIdTransformation,
    SerializeRawResponseTransformation,
)
from chess_teacher.pipelines.modes import (
    PipelineMode,
    StorageFolder,
    ingestion_load_merge_strategy,
    ingestion_load_source_folders,
)
from chess_teacher.platform.account import Account
from chess_teacher.utils.db.client import DatabaseClient
from chess_teacher.utils.exception_utils import (
    AdapterError,
    DatabaseError,
    PipelineError,
)
from chess_teacher.utils.files.file_utils import FileType
from chess_teacher.utils.files.file_writer import FileWriter, FileWriterFactory
from chess_teacher.utils.general_utils import (
    build_daily_key,
    generate_ident_is_literal,
    get_current_datetime,
)
from chess_teacher.utils.object_storage.base import ObjectStorage
from chess_teacher.utils.object_storage.factory import get_raw_storage
from chess_teacher.utils.pipeline_utils.pipeline_base import PipelineContext, PipelineStep
from chess_teacher.utils.pipeline_utils.pipeline_steps import (
    LoadingStrategy,
    LoadToDatabaseStep,
    StorageToTableStep,
)
from chess_teacher.utils.pipeline_utils.transformations import (
    CreateHashedIdTransformation,
    JoinWithTableTransformation,
)

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
    """Load JSONL objects from storage into games.raw_games and archive them by outcome."""

    def __init__(
        self,
        *,
        mode: PipelineMode = PipelineMode.INCREMENTAL,
        storage: ObjectStorage | None = None,
    ) -> None:
        super().__init__(
            name="LoadIngestedFilesToDB",
            storage_path="",
            file_type=_INGESTION_FILE_TYPE,
            data_class=RawGame,
            storage=storage,
            transformations=[
                SerializeRawResponseTransformation(),
                ExtractFileMetadataTransformation(),
                JoinWithTableTransformation(with_data_class=Account),
                ExtractPlatformGameIdTransformation(),
                CreateHashedIdTransformation(data_class=RawGame),
            ],
            loading_strategy=LoadingStrategy.MERGE,
            merge_strategy=ingestion_load_merge_strategy(mode),
        )
        self.mode = mode
        self._account: Account | None = None
        self._ingested_prefix = ""
        self._failed_prefix = ""
        self._processed_prefix = ""
        self._key_source_folder: dict[str, StorageFolder] = {}

    def _account_prefix(self, folder: StorageFolder) -> str:
        assert self._account is not None
        return _get_account_storage_prefix(folder, self._account)

    def _resolve_storage_paths(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        account = _fetch_account(db_client, context)
        self._account = account
        self._ingested_prefix = self._account_prefix("ingested")
        self._failed_prefix = self._account_prefix("failed")
        self._processed_prefix = self._account_prefix("processed")
        self.storage_path = self._ingested_prefix
        self.quarantine_path = self._failed_prefix
        if ingestion_load_merge_strategy(self.mode).when_not_matched_by_source == "delete":
            self.match_condition = generate_ident_is_literal("account_id", account.account_id)

    def _list_source_keys(self) -> list[str]:
        storage = self._get_storage()
        keys: list[str] = []
        seen: set[str] = set()
        for folder in ingestion_load_source_folders(self.mode):
            prefix = self._account_prefix(folder)
            folder_keys = storage.list_keys(
                prefix,
                recursive=self.recursive,
                suffix=self.file_type.value,
                glob_pattern=self.glob_pattern,
            )
            for key in folder_keys:
                if key in seen:
                    continue
                seen.add(key)
                keys.append(key)
                self._key_source_folder[key] = folder
        return keys

    def _relative_under_folder(self, key: str, folder: StorageFolder) -> str:
        return ObjectStorage.relative_key_under(key, self._account_prefix(folder))

    def _destination_key(self, key: str, dest_folder: StorageFolder) -> str:
        source_folder = self._key_source_folder[key]
        relative = self._relative_under_folder(key, source_folder)
        destination = ObjectStorage.resolve_key(self._account_prefix(dest_folder), relative)
        storage = self._get_storage()
        if destination != key and storage.read_bytes(destination) is not None:
            destination = ObjectStorage.resolve_key(
                self._account_prefix(dest_folder),
                ObjectStorage.unique_key_variant(relative, uuid4().hex),
            )
        return destination

    def _move_key(self, key: str, dest_folder: StorageFolder) -> None:
        destination = self._destination_key(key, dest_folder)
        if key == destination:
            return
        storage = self._get_storage()
        try:
            storage.move_verified(key, destination, overwrite=False)
            self.logger.info(f"[{self.name}] Moved {key} -> {destination}.")
            self._key_source_folder[destination] = dest_folder
            self._key_source_folder.pop(key, None)
        except Exception as e:
            self.logger.error(f"[{self.name}] Failed to move {key} to {dest_folder}: {e}")

    def _move_keys(self, keys: list[str], dest_folder: StorageFolder) -> None:
        for key in keys:
            self._move_key(key, dest_folder)

    def _ensure_ingested_empty(self) -> None:
        storage = self._get_storage()
        leftover = storage.list_keys(
            self._ingested_prefix,
            recursive=self.recursive,
            suffix=self.file_type.value,
            glob_pattern=self.glob_pattern,
        )
        if not leftover:
            return
        self.logger.warning(
            f"[{self.name}] {len(leftover)} object(s) still under {self._ingested_prefix}; "
            "moving to failed."
        )
        for key in leftover:
            self._key_source_folder[key] = "ingested"
            self._move_key(key, "failed")

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        self._resolve_storage_paths(db_client, context)
        self._key_source_folder = {}
        self._loaded_keys = []
        try:
            LoadToDatabaseStep.run(self, db_client, context)
            if self._loaded_keys:
                self._move_keys(self._loaded_keys, "processed")
        except Exception:
            if self._loaded_keys:
                self._move_keys(self._loaded_keys, "failed")
            raise
        finally:
            self._ensure_ingested_empty()

    def _load_records(self, db_client: DatabaseClient, context: PipelineContext) -> pl.DataFrame:
        """Load records from configured storage folders into a Polars DataFrame."""
        storage = self._get_storage()
        keys = self._list_source_keys()

        if not keys:
            folders = ", ".join(ingestion_load_source_folders(self.mode))
            self.logger.warning(
                f"[{self.name}] No objects found for mode={self.mode!r} "
                f"(folders={folders}, suffix=.{self.file_type.value})."
            )
            context.progress_pop()
            context.progress_warning("No files found to extract records from.")
            return pl.DataFrame()

        file_total = len(keys)
        context.progress_update(f"Found {file_total} file{'s' if file_total != 1 else ''} to load.")
        records: list[dict] = []
        for file_index, key in enumerate(keys, start=1):
            context.progress_update(f"Loading file {file_index}/{file_total}...")
            self.logger.info(f"[{self.name}] Loading {key}.")
            try:
                file_records = self.file_loader.load_key(storage, key)
            except Exception as e:
                self.logger.warning(f"[{self.name}] Failed to load {key}: {e}")
                self._move_key(key, "failed")
                continue
            self.logger.info(f"[{self.name}] Loaded {len(file_records)} records from {key}.")

            try:
                ingestion_ts = get_current_datetime()
                for record in file_records:
                    record["_source_file"] = key
                    record["_ingestion_ts"] = ingestion_ts
                records.extend(file_records)
                self._loaded_keys.append(key)
                context.loaded_storage_keys.append(key)
                self.logger.info(f"[{self.name}] Added metadata to {key}.")
            except Exception as e:
                self.logger.warning(f"[{self.name}] Failed to add metadata to {key}: {e}")
                self._move_key(key, "failed")
        self.logger.info(f"[{self.name}] Loaded {len(records)} records from {len(keys)} keys.")
        context.progress_update(
            f"Loaded {len(records)} record{'s' if len(records) != 1 else ''}. "
            f"Processed {len(self._loaded_keys)}/{len(keys)} file{'s' if len(self._loaded_keys) != 1 else ''} successfully."
        )

        return pl.DataFrame(records)
