from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from chess_teacher.ingestion.adapter import AdapterFactory
from chess_teacher.pipelines.pipeline_base import PipelineStep
from chess_teacher.pipelines.pipeline_steps import StorageToTableStep
from chess_teacher.platform.account import Account
from chess_teacher.utils.db_client import DatabaseClient
from chess_teacher.utils.env_utils import get_env_variable
from chess_teacher.utils.exception_utils import (
    AdapterError,
    ConfigError,
    DatabaseError,
    FileError,
)
from chess_teacher.utils.file_utils import FileType, discover_files, move_file
from chess_teacher.utils.file_writer import FileWriter, FileWriterFactory
from chess_teacher.utils.general_utils import build_daily_path, get_current_datetime

_INGESTION_FILE_TYPE = FileType.JSONL


def _get_target_base_path(
    folder: Literal["ingested", "failed", "processed"], account: Account
) -> Path:
    try:
        raw_dir = get_env_variable("RAW_DIR")
        if not raw_dir:
            raise ConfigError("RAW_DIR environment variable is not set")
        result = build_daily_path(Path(raw_dir) / folder / account.account_id)
    except ValueError as e:
        raise ConfigError(f"RAW_DIR environment variable is not set: {e}")
    return result


class IngestionFromAPIStreamStep(PipelineStep):
    """Ingest data from an API stream into a storage location."""

    def __init__(self, account: Account):
        self.account = account
        name = f"APIStreamIngestion_{account.account_id}"
        super().__init__(name=name)  # also sets self.logger

        self.target_base_path = _get_target_base_path("ingested", account)
        self.adapter = AdapterFactory.from_account(account)
        self._writer: FileWriter = FileWriterFactory.get_writer(
            _INGESTION_FILE_TYPE, logger=self.logger
        )

    def _generate_filename(self) -> str:
        """Generate the name of the output file."""
        return f"{self.account.platform.value}_{uuid4().hex}.{_INGESTION_FILE_TYPE.value}"

    def _get_last_updated(self, db_client: DatabaseClient) -> datetime | None:
        """Fetch the last updated time from the database to get the most up-to-date value."""
        try:
            result = self.account.fetch_from_db(
                db_client, id=self.account.account_id
            ).latest_ingestion
        except Exception as e:
            self.logger.log_and_raise(DatabaseError(f"Error fetching last updated time: {e}"))
        return result

    def _set_last_updated(
        self, db_client: DatabaseClient, last_updated: datetime = get_current_datetime()
    ) -> None:
        """Set the last updated time in the database."""
        try:
            self.account.upsert_latest(db_client, "latest_ingestion", last_updated)
        except Exception as e:
            self.logger.log_and_raise(DatabaseError(f"Error setting last updated time: {e}"))

    def run(self, db_client: DatabaseClient) -> None:
        output_path = self.target_base_path / self._generate_filename()
        since = self._get_last_updated(db_client)
        since_new = get_current_datetime()

        try:
            records = self.adapter.get_records(since=since)
            if not records:
                self.logger.info(f"[{self.name}] No records to write.")
                return
        except Exception as e:
            self.logger.log_and_raise(AdapterError(f"Error getting records: {e}"))

        self._writer.write(records, output_path)
        self.logger.info(f"[{self.name}] Written to {output_path}.")
        self._set_last_updated(db_client, since_new)
        self.logger.info(f"[{self.name}] Ingestion completed.")


# CONTINUE: CREATE TABLE, TRANSFORMATION, LOADING STRATEGY, METADATA, AND THE FOLLOWING PIPELINE STEP TO ADD TO PIPELINE
class LoadIngestedFilesToDB(StorageToTableStep):
    """Load ingested files to the database."""

    def __init__(self, account: Account):
        self.account = account
        name = f"LoadIngestedFilesToDB_{account.account_id}"
        super().__init__(
            name=name,
            storage_path=_get_target_base_path("ingested", account),
            file_type=_INGESTION_FILE_TYPE,
            # table_metadata=self.table_metadata, # TODO: create table
            # transformations=self.transformations,
            # loading_strategy=self.loading_strategy,
            # merge_strategy=self.merge_strategy,
            # cascade=self.cascade,
            # match_condition=self.match_condition
        )


class ArchiveIngestedFilesStep(PipelineStep):
    """
    Move successfully processed ingested files from source storage to archive storage.

    Intended to run after ``StorageToTableStep`` in the same pipeline so only files
    that were loaded into the database (and not quarantined) remain under ``source_path``.
    """

    file_type: FileType = FileType.JSONL

    def __init__(
        self,
        account: Account,
        *,
        recursive: bool = True,
        glob_pattern: str | None = None,
    ) -> None:
        super().__init__(name=f"ArchiveIngestedFiles_{account.account_id}")
        self.account = account
        self.source_path = _get_target_base_path("ingested", account)
        self.archive_path = _get_target_base_path("processed", account)
        self.recursive = recursive
        self.glob_pattern = glob_pattern

    def _archive_destination(self, source: Path) -> Path:
        """Preserve relative layout under source_path in the archive."""
        if self.recursive:
            relative = source.relative_to(self.source_path)
        else:
            relative = Path(source.name)
        destination = self.archive_path / relative
        if not destination.exists():
            return destination
        return destination.with_name(f"{destination.stem}_{uuid4().hex}{destination.suffix}")

    def run(self, db_client: DatabaseClient) -> None:
        paths = discover_files(
            self.source_path,
            recursive=self.recursive,
            suffix=self.file_type.value,
            glob_pattern=self.glob_pattern,
            logger=self.logger,
        )

        if not paths:
            self.logger.info(f"[{self.name}] No files to archive under {self.source_path}.")
            return

        archived = 0
        for path in paths:
            destination = self._archive_destination(path)
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

        self.logger.info(f"[{self.name}] Archived {archived} file(s) to {self.archive_path}.")
