from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from chess_teacher.ingestion.adapter import AdapterFactory
from chess_teacher.ingestion.raw_games import RawGame
from chess_teacher.ingestion.transformations import (
    ExtractFileMetadataTransformation,
    ExtractGameMetadataTransformation,
    ExtractPlatformGameIdTransformation,
    ExtractPlayersAndResultTransformation,
    FilterGamesWithPGNTransformation,
)
from chess_teacher.pipelines.pipeline_base import PipelineContext, PipelineStep
from chess_teacher.pipelines.pipeline_steps import (
    LoadingStrategy,
    MergeStrategy,
    StorageToTableStep,
)
from chess_teacher.pipelines.transformations import (
    CreateHashedIdTransformation,
    JoinWithTableTransformation,
)
from chess_teacher.platform.account import Account
from chess_teacher.utils.db_client import DatabaseClient
from chess_teacher.utils.env_utils import get_env_variable
from chess_teacher.utils.exception_utils import (
    AdapterError,
    ConfigError,
    DatabaseError,
    FileError,
    PipelineError,
)
from chess_teacher.utils.file_utils import FileType, discover_files, move_file
from chess_teacher.utils.file_writer import FileWriter, FileWriterFactory
from chess_teacher.utils.general_utils import build_daily_path, get_current_datetime

_INGESTION_FILE_TYPE = FileType.JSONL


def _get_account_storage_path(
    folder: Literal["ingested", "failed", "processed"], account: Account
) -> Path:
    """Account-level storage root (all dates under YYYY/MM/DD subdirs)."""
    try:
        raw_dir = get_env_variable("RAW_DIR")
        if not raw_dir:
            raise ConfigError("RAW_DIR environment variable is not set")
        path = Path(raw_dir) / folder / account.account_id
        path.mkdir(parents=True, exist_ok=True)
        return path
    except ValueError as e:
        raise ConfigError(f"RAW_DIR environment variable is not set: {e}")


def _get_daily_ingest_path(account: Account) -> Path:
    """Today's ingest directory where API stream writes new files."""
    return build_daily_path(_get_account_storage_path("ingested", account))


def _fetch_account(db_client: DatabaseClient, context: PipelineContext) -> Account:
    if context.account_id is None:
        raise PipelineError("account_id is required for ingestion pipeline steps")
    return Account.fetch_from_db(db_client, id=context.account_id)


class IngestionFromAPIStreamStep(PipelineStep):
    """Ingest data from an API stream into a storage location."""

    def __init__(self) -> None:
        super().__init__(name="IngestionFromAPIStream")

    def _generate_filename(self, account: Account) -> str:
        """Generate the name of the output file."""
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
        target_base_path = _get_daily_ingest_path(account)
        adapter = AdapterFactory.from_account(account)
        writer: FileWriter = FileWriterFactory.get_writer(_INGESTION_FILE_TYPE, logger=self.logger)

        output_path = target_base_path / self._generate_filename(account)
        since = self._get_last_updated(db_client, account)
        since_new = get_current_datetime()

        try:
            records = adapter.get_records(since=since)
            if not records:
                self.logger.info(f"[{self.name}] No records to write.")
                return
        except Exception as e:
            self.logger.log_and_raise(AdapterError(f"Error getting records: {e}"))

        writer.write(records, output_path)
        self.logger.info(f"[{self.name}] Written to {output_path}.")
        self._set_last_updated(db_client, account, since_new)
        self.logger.info(f"[{self.name}] Ingestion completed.")


class LoadIngestedFilesToDB(StorageToTableStep):
    """Load ingested files to the database."""

    def __init__(self) -> None:
        super().__init__(
            name="LoadIngestedFilesToDB",
            storage_path="",
            file_type=_INGESTION_FILE_TYPE,
            data_class=RawGame,
            transformations=[
                FilterGamesWithPGNTransformation(),
                ExtractFileMetadataTransformation(),
                JoinWithTableTransformation(with_data_class=Account),
                ExtractPlatformGameIdTransformation(),
                CreateHashedIdTransformation(data_class=RawGame),
                ExtractGameMetadataTransformation(),
                ExtractPlayersAndResultTransformation(),
            ],
            loading_strategy=LoadingStrategy.MERGE,
            merge_strategy=MergeStrategy.upsert(),
        )

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        account = _fetch_account(db_client, context)
        self.storage_path = str(_get_account_storage_path("ingested", account))
        super().run(db_client, context)


class ArchiveIngestedFilesStep(PipelineStep):
    """
    Move successfully processed ingested files from source storage to archive storage.

    Intended to run after ``StorageToTableStep`` in the same pipeline so only files
    that were loaded into the database (and not quarantined) remain under ``source_path``.
    """

    file_type: FileType = FileType.JSONL

    def __init__(
        self,
        *,
        recursive: bool = True,
        glob_pattern: str | None = None,
    ) -> None:
        super().__init__(name="ArchiveIngestedFiles")
        self.recursive = recursive
        self.glob_pattern = glob_pattern

    def _archive_destination(self, source: Path, source_path: Path, archive_path: Path) -> Path:
        """Preserve relative layout under source_path in the archive."""
        if self.recursive:
            relative = source.relative_to(source_path)
        else:
            relative = Path(source.name)
        destination = archive_path / relative
        if not destination.exists():
            return destination
        return destination.with_name(f"{destination.stem}_{uuid4().hex}{destination.suffix}")

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        account = _fetch_account(db_client, context)
        source_path = _get_account_storage_path("ingested", account)
        archive_path = _get_account_storage_path("processed", account)

        paths = discover_files(
            source_path,
            recursive=self.recursive,
            suffix=self.file_type.value,
            glob_pattern=self.glob_pattern,
            logger=self.logger,
        )

        if not paths:
            self.logger.info(f"[{self.name}] No files to archive under {source_path}.")
            return

        archived = 0
        for path in paths:
            destination = self._archive_destination(path, source_path, archive_path)
            try:
                move_file(
                    path,
                    destination,
                    overwrite=False,
                    mkdir=True,
                    logger=self.logger,
                )
            except FileError as e:
                self.logger.log_and_raise(
                    FileError(f"Failed to archive {path} to {destination}: {e}")
                )
            self.logger.info(f"[{self.name}] Archived {path} -> {destination}.")
            archived += 1

        self.logger.info(f"[{self.name}] Archived {archived} file(s) to {archive_path}.")
