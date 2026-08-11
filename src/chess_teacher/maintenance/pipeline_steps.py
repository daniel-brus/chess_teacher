from datetime import date, timedelta
from pathlib import PurePosixPath

import polars as pl

from chess_teacher.maintenance.dataclasses import (
    ExceptionHourlyCount,
    LogLevelHourlyCount,
    RawLog,
    WarningErrorLog,
)
from chess_teacher.maintenance.transformations import (
    AggregateExceptionHourlyTransformation,
    AggregateLogLevelHourlyTransformation,
    FilterWarningErrorLevelsTransformation,
    ParseLogTimestampColumnsTransformation,
)
from chess_teacher.utils.cache_utils import invalidate_admin_log_aggregates_cache
from chess_teacher.utils.db.client import DatabaseClient, MergeStrategy
from chess_teacher.utils.files.file_utils import FileType
from chess_teacher.utils.general_utils import get_current_datetime
from chess_teacher.utils.metadata_utils import TableMetadata
from chess_teacher.utils.object_storage.base import ObjectStorage
from chess_teacher.utils.object_storage.factory import get_raw_storage
from chess_teacher.utils.pipeline_utils.pipeline_base import (
    PipelineContext,
    PipelineRunResult,
    PipelineStep,
)
from chess_teacher.utils.pipeline_utils.pipeline_steps import (
    LoadingStrategy,
    LoadToDatabaseStep,
    StorageToTableStep,
    TransformStep,
)
from chess_teacher.utils.pipeline_utils.transformations import (
    CastToDatetimeTransformation,
    RenameColumnsTransformation,
)

_RAW_LOG_RETENTION = timedelta(days=30)
_WARNING_ERROR_LOG_RETENTION = timedelta(days=365)
_S3_LOG_RETENTION = timedelta(days=30)
_LOG_AGGREGATION_WINDOW = timedelta(days=7)


class DeleteOldRecordsStep(PipelineStep):
    def __init__(
        self,
        name: str,
        metadata: TableMetadata,
        column: str,
        *,
        retention_period: timedelta,
        additional_where: str | None = None,
    ):
        super().__init__(name, critical=False)
        self.metadata = metadata
        self.column = column
        self.retention_period = retention_period
        self.additional_where = additional_where

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        """Delete old records from the table"""
        cutoff_datetime = get_current_datetime() - self.retention_period
        where = f"""{self.column} < '{cutoff_datetime.isoformat()}'""" + (
            f" AND ({self.additional_where})" if self.additional_where else ""
        )
        rows_deleted = db_client.delete_where(self.metadata, where=where)
        context.progress_update(
            f"Deleted {rows_deleted} old record{'s' if rows_deleted != 1 else ''}."
        )
        self.logger.info(
            f"Deleted {rows_deleted} old records from {self.metadata.qualified_name_sql()} ({where})"
        )
        return


class LoadRawLogsStep(StorageToTableStep):
    """Load shipped JSON-lines log segments from object storage into logs.raw_logs.

    Segments under ``closed/`` are pending ingest. After a successful DB load they
    move to ``processed/`` so the next nightly run does not re-download the backlog
    (Backblaze Class B / download transactions). Retention cleanup lives in
    :class:`DeleteOldS3LogFilesStep`.
    """

    CLOSED_LOG_STORAGE_PREFIX = "logs/python/buffer/closed"
    PROCESSED_LOG_STORAGE_PREFIX = "logs/python/buffer/processed"
    QUARANTINE_LOG_STORAGE_PREFIX = "logs/python/quarantine"
    LOG_FILE_SUFFIX = "log"
    _UNKNOWN_HOSTNAME = "unknown"

    def __init__(self, *, storage: ObjectStorage | None = None) -> None:
        super().__init__(
            name="load_raw_logs",
            storage_path=self.CLOSED_LOG_STORAGE_PREFIX,
            file_type=FileType.JSONL,
            data_class=RawLog,
            storage=storage,
            transformations=[
                RenameColumnsTransformation({
                    "_source_file": "source_file",
                    "_ingestion_ts": "loaded_at",
                }),
                ParseLogTimestampColumnsTransformation(),
                CastToDatetimeTransformation(columns=["loaded_at"]),
            ],
            quarantine_path=self.QUARANTINE_LOG_STORAGE_PREFIX,
            loading_strategy=LoadingStrategy.INSERT_IGNORE,
        )

    @staticmethod
    def relative_closed_log_key(source_file: str) -> str:
        """Return the path under ``CLOSED_LOG_STORAGE_PREFIX`` for a storage key."""
        prefix = f"{LoadRawLogsStep.CLOSED_LOG_STORAGE_PREFIX}/"
        if source_file.startswith(prefix):
            return source_file[len(prefix) :]
        return source_file

    @classmethod
    def processed_key_for_closed(cls, closed_key: str) -> str:
        """Map a ``closed/`` key to the matching ``processed/`` key."""
        relative = ObjectStorage.relative_key_under(closed_key, cls.CLOSED_LOG_STORAGE_PREFIX)
        return ObjectStorage.resolve_key(cls.PROCESSED_LOG_STORAGE_PREFIX, relative)

    @staticmethod
    def parse_closed_log_hostname(source_file: str) -> str:
        """
        Parse hostname/pod from a closed log segment storage key.

        Expected layout: ``.../closed/{YYYY}/{MM}/{DD}/{hostname}/{segment}.log``
        """
        relative = LoadRawLogsStep.relative_closed_log_key(source_file)
        if DeleteOldS3LogFilesStep.parse_log_path_date(relative) is None:
            return LoadRawLogsStep._UNKNOWN_HOSTNAME
        parts = PurePosixPath(relative).parts
        if len(parts) >= 4:
            return parts[3]
        return LoadRawLogsStep._UNKNOWN_HOSTNAME

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        """Load into DB, then move successfully ingested segments to ``processed/``."""
        self._resolve_storage_paths(db_client, context)
        self._loaded_keys = []
        try:
            LoadToDatabaseStep.run(self, db_client, context)
            self._move_keys_to_processed(self._loaded_keys, context)
        except Exception:
            self._quarantine_keys(self._loaded_keys)
            raise

    def _already_loaded_source_files(self, db_client: DatabaseClient) -> set[str]:
        """Return ``source_file`` values already present in ``logs.raw_logs``."""
        meta = RawLog.get_metadata()
        if not db_client.table_exists(meta):
            return set()
        sql = f"SELECT DISTINCT source_file FROM {meta.qualified_name_sql()};"
        rows = db_client.engine.execute_parameterized_query(sql, {})
        return {row["source_file"] for row in rows if row.get("source_file")}

    def _move_keys_to_processed(self, keys: list[str], context: PipelineContext) -> None:
        if not keys:
            return
        storage = self._get_storage()
        moved = 0
        for key in keys:
            destination = self.processed_key_for_closed(key)
            try:
                # overwrite=True skips HeadObject (Class B) on the destination.
                storage.move(key, destination, overwrite=True)
                moved += 1
                self.logger.info(f"[{self.name}] Moved {key} -> {destination}.")
            except Exception as e:
                self.logger.error(f"[{self.name}] Failed to move {key} to processed: {e}")
        if moved:
            context.progress_update(
                f"Moved {moved} ingested log segment{'s' if moved != 1 else ''} "
                f"to {self.PROCESSED_LOG_STORAGE_PREFIX}."
            )
            self.logger.info(
                f"[{self.name}] Moved {moved} ingested log segment(s) to "
                f"{self.PROCESSED_LOG_STORAGE_PREFIX}."
            )

    def _load_records(self, db_client: DatabaseClient, context: PipelineContext) -> pl.DataFrame:
        storage = self._get_storage()
        keys = storage.list_keys(
            self.storage_path,
            recursive=self.recursive,
            suffix=self.LOG_FILE_SUFFIX,
            glob_pattern=self.glob_pattern,
        )

        if not keys:
            self.logger.warning(
                f"[{self.name}] No objects found at {self.storage_path} "
                f"(recursive={self.recursive}, suffix=.{self.LOG_FILE_SUFFIX})."
            )
            context.progress_pop()
            context.progress_warning("No log files found to load.")
            return pl.DataFrame()

        already_loaded = self._already_loaded_source_files(db_client)
        stale_keys = [key for key in keys if key in already_loaded]
        if stale_keys:
            # Archive without downloading — avoids thousands of Class B GETs each night.
            self._move_keys_to_processed(stale_keys, context)
            keys = [key for key in keys if key not in already_loaded]

        if not keys:
            context.progress_pop()
            context.progress_success("No new log files to load.")
            return pl.DataFrame()

        file_total = len(keys)
        context.progress_update(
            f"Found {file_total} new log file{'s' if file_total != 1 else ''} to load."
        )
        records: list[dict] = []
        for file_index, key in enumerate(keys, start=1):
            context.progress_update(f"Loading log file {file_index}/{file_total}...")
            self.logger.info(f"[{self.name}] Loading {key}.")
            try:
                file_records = self.file_loader.load_key(storage, key)
            except Exception as e:
                self.logger.warning(f"[{self.name}] Failed to load {key}: {e}")
                self._quarantine_keys([key])
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
            except Exception as e:
                self.logger.warning(f"[{self.name}] Failed to add metadata to {key}: {e}")
                self._quarantine_keys([key])

        self.logger.info(f"[{self.name}] Loaded {len(records)} records from {len(keys)} keys.")
        context.progress_update(
            f"Loaded {len(records)} log record{'s' if len(records) != 1 else ''}. "
            f"Processed {len(self._loaded_keys)}/{len(keys)} file"
            f"{'s' if len(self._loaded_keys) != 1 else ''} successfully."
        )
        return pl.DataFrame(records)


class PromoteWarningErrorLogsStep(TransformStep):
    """Copy WARNING, ERROR, and CRITICAL rows from raw_logs into warning_error_logs."""

    def __init__(self) -> None:
        super().__init__(
            name="promote_warning_error_logs",
            source_data_class=RawLog,
            target_data_class=WarningErrorLog,
            on="log_id",
            transformations=[FilterWarningErrorLevelsTransformation()],
            loading_strategy=LoadingStrategy.INSERT_IGNORE,
        )


class AggregateLogLevelHourlyStep(TransformStep):
    """Aggregate raw_logs from the last 7 days into hourly level counts."""

    def __init__(self) -> None:
        super().__init__(
            name="aggregate_log_level_hourly_counts",
            source_data_class=RawLog,
            target_data_class=LogLevelHourlyCount,
            on=None,
            transformations=[AggregateLogLevelHourlyTransformation()],
            loading_strategy=LoadingStrategy.MERGE,
            merge_strategy=MergeStrategy.upsert(),
        )

    def _load_records(self, db_client: DatabaseClient, context: PipelineContext) -> pl.DataFrame:
        source = self.source_table_metadata.qualified_name_sql()
        if not db_client.table_exists(self.source_table_metadata):
            self.logger.warning(
                f"[{self.name}] Source table {source} does not exist; using empty frame."
            )
            return pl.DataFrame()
        window_start = get_current_datetime() - _LOG_AGGREGATION_WINDOW
        where = f"ts >= '{window_start.isoformat()}'"
        context.progress_update(f"Reading raw logs since {window_start.isoformat()}...")
        return db_client.read(
            self.source_table_metadata,
            columns=self.source_columns,
            where=where,
            as_polars=True,
        )


class AggregateExceptionHourlyStep(TransformStep):
    """Aggregate warning/error logs from the last 7 days into hourly counts by level and exc_type."""

    def __init__(self) -> None:
        super().__init__(
            name="aggregate_exception_hourly_counts",
            source_data_class=WarningErrorLog,
            target_data_class=ExceptionHourlyCount,
            on=None,
            transformations=[AggregateExceptionHourlyTransformation()],
            loading_strategy=LoadingStrategy.MERGE,
            merge_strategy=MergeStrategy.upsert(),
        )

    def _load_records(self, db_client: DatabaseClient, context: PipelineContext) -> pl.DataFrame:
        source = self.source_table_metadata.qualified_name_sql()
        if not db_client.table_exists(self.source_table_metadata):
            self.logger.warning(
                f"[{self.name}] Source table {source} does not exist; using empty frame."
            )
            return pl.DataFrame()
        window_start = get_current_datetime() - _LOG_AGGREGATION_WINDOW
        where = f"ts >= '{window_start.isoformat()}'"
        context.progress_update(f"Reading warning/error logs since {window_start.isoformat()}...")
        return db_client.read(
            self.source_table_metadata,
            columns=self.source_columns,
            where=where,
            as_polars=True,
        )


class DeleteOldRawLogsStep(DeleteOldRecordsStep):
    def __init__(self) -> None:
        super().__init__(
            "delete_old_raw_logs",
            RawLog.get_metadata(),
            "ts",
            retention_period=_RAW_LOG_RETENTION,
        )


class DeleteOldWarningErrorLogsStep(DeleteOldRecordsStep):
    def __init__(self) -> None:
        super().__init__(
            "delete_old_warning_error_logs",
            WarningErrorLog.get_metadata(),
            "ts",
            retention_period=_WARNING_ERROR_LOG_RETENTION,
        )


class DeleteOldS3LogFilesStep(PipelineStep):
    """Delete log segments older than the retention window (by path date).

    Cleans ``processed/`` (normal post-ingest archive) and ``closed/`` (orphans
    that never loaded).
    """

    PROCESSED_LOG_STORAGE_PREFIX = LoadRawLogsStep.PROCESSED_LOG_STORAGE_PREFIX
    CLOSED_LOG_STORAGE_PREFIX = LoadRawLogsStep.CLOSED_LOG_STORAGE_PREFIX
    LOG_FILE_SUFFIX = "log"

    def __init__(self, *, storage: ObjectStorage | None = None) -> None:
        super().__init__("delete_old_s3_log_files", critical=False)
        self._storage = storage

    @staticmethod
    def parse_log_path_date(relative_key: str) -> date | None:
        """
        Parse the log segment date from a key relative to a buffer folder prefix.

        Expected layout: ``{YYYY}/{MM}/{DD}/{hostname}/{segment}.log``
        """
        parts = PurePosixPath(relative_key).parts
        if len(parts) < 4:
            return None
        try:
            return date(int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError:
            return None

    @classmethod
    def parse_closed_log_path_date(cls, relative_key: str) -> date | None:
        """Alias for :meth:`parse_log_path_date` (kept for existing call sites)."""
        return cls.parse_log_path_date(relative_key)

    def _get_storage(self) -> ObjectStorage:
        return self._storage if self._storage is not None else get_raw_storage()

    def _keys_past_retention(
        self, storage: ObjectStorage, prefix: str, cutoff_date: date
    ) -> list[str]:
        keys = storage.list_keys(prefix, recursive=True, suffix=self.LOG_FILE_SUFFIX)
        to_delete: list[str] = []
        for key in keys:
            relative = ObjectStorage.relative_key_under(key, prefix)
            path_date = self.parse_log_path_date(relative)
            if path_date is None:
                self.logger.warning(f"[{self.name}] Skipping key with unparseable path date: {key}")
                continue
            if path_date < cutoff_date:
                to_delete.append(key)
        return to_delete

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        storage = self._get_storage()
        cutoff_date = (get_current_datetime() - _S3_LOG_RETENTION).date()
        to_delete = self._keys_past_retention(
            storage, self.PROCESSED_LOG_STORAGE_PREFIX, cutoff_date
        )
        to_delete.extend(
            self._keys_past_retention(storage, self.CLOSED_LOG_STORAGE_PREFIX, cutoff_date)
        )

        if not to_delete:
            context.progress_update("No old S3 log files to delete.")
            self.logger.info(f"[{self.name}] No S3 log files older than {cutoff_date}.")
            return

        storage.delete_keys(to_delete)
        context.progress_update(
            f"Deleted {len(to_delete)} old S3 log file{'s' if len(to_delete) != 1 else ''}."
        )
        self.logger.info(
            f"[{self.name}] Deleted {len(to_delete)} S3 log file(s) with path date before {cutoff_date}."
        )


class InvalidateAdminLogDashboardCacheStep(PipelineStep):
    """Drop Redis cache for admin log dashboard aggregate tables."""

    def __init__(self) -> None:
        super().__init__("invalidate_admin_log_dashboard_cache", critical=False)

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        invalidate_admin_log_aggregates_cache()
        context.progress_update("Admin log dashboard cache invalidated.")
        self.logger.info("[%s] Invalidated admin log dashboard Redis cache.", self.name)


class ClearOrphanedPipelineRunLocksStep(DeleteOldRecordsStep):
    """Clear orphaned pipeline run locks"""

    _RETENTION_PERIOD: timedelta = timedelta(days=1)

    def __init__(self):
        super().__init__(
            "clear_orphaned_pipeline_run_locks",
            PipelineRunResult.get_metadata(),
            "started_at",
            retention_period=self._RETENTION_PERIOD,
            additional_where="finished_at = '1970-01-01 00:00:00+00'",
        )
