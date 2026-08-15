from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import ClassVar
from uuid import uuid4

import polars as pl

from chess_teacher.utils.db.client import DatabaseClient, MergeStrategy, WriteResult
from chess_teacher.utils.exception_utils import MetadataError, PipelineError
from chess_teacher.utils.files.file_loader import FileLoader, FileLoaderFactory
from chess_teacher.utils.files.file_utils import FileType, TextStreamSource
from chess_teacher.utils.general_utils import (
    generate_ident_is_literal,
    get_current_datetime,
    quote_ident,
)
from chess_teacher.utils.metadata_utils import TableMetadata
from chess_teacher.utils.object_storage.base import ObjectStorage
from chess_teacher.utils.object_storage.factory import get_raw_storage
from chess_teacher.utils.pipeline_utils.pipeline_base import PipelineContext, PipelineStep
from chess_teacher.utils.pipeline_utils.transformations import (
    AssertUniqueColumnsTransformation,
    CastDataTypeTransformation,
    CastToDatetimeTransformation,
    DataFrameTransformation,
    FilterColumnsTransformation,
    IncrementalFilterTransformation,
)
from chess_teacher.utils.table_data_class import TableDataClass


class LoadingStrategy(StrEnum):
    APPEND = "append"
    INSERT_IGNORE = "insert_ignore"
    OVERWRITE = "overwrite"
    MERGE = "merge"


MetadataTransformationFactory = Callable[[type[TableDataClass]], DataFrameTransformation]


class LoadToDatabaseStep(PipelineStep):
    """Load data from arbitrary source into a table."""

    DEFAULT_TRANSFORMATIONS: ClassVar[list[MetadataTransformationFactory]] = [
        CastDataTypeTransformation,
        FilterColumnsTransformation,
    ]

    def __init__(
        self,
        name: str,
        data_class: type[TableDataClass],
        transformations: list[DataFrameTransformation] = [],
        *,
        loading_strategy: LoadingStrategy,
        merge_strategy: MergeStrategy | None = None,
        cascade: bool | None = None,
        match_condition: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.data_class = data_class
        self.table_metadata = data_class.get_metadata()

        # apply metadata-dependent transformations after the user-provided transformations
        default_transformations: list[DataFrameTransformation] = [
            transformation(data_class) for transformation in self.DEFAULT_TRANSFORMATIONS
        ]
        primary_key = self.table_metadata.primary_key
        if primary_key:
            default_transformations.append(
                AssertUniqueColumnsTransformation(
                    list(primary_key),
                    label=", ".join(primary_key),
                )
            )
        self.transformations = transformations + default_transformations

        self.loading_strategy = loading_strategy
        # load strategy-specific configurations
        if loading_strategy == LoadingStrategy.MERGE:
            self.merge_strategy = merge_strategy or MergeStrategy.upsert()
            self.match_condition = match_condition
        elif loading_strategy == LoadingStrategy.OVERWRITE:
            self.cascade = cascade

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        table = self.table_metadata.qualified_name_sql()
        self.logger.info(
            f"[{self.name}] Loading into {table} (strategy={self.loading_strategy.value})."
        )
        if self.loading_strategy == LoadingStrategy.MERGE:
            merge_label = self.merge_strategy
            self.logger.info(
                f"[{self.name}] Merge strategy: "
                f"matched={merge_label.when_matched}, "
                f"not_matched_by_target={merge_label.when_not_matched_by_target}, "
                f"not_matched_by_source={merge_label.when_not_matched_by_source}."
            )

        db_client.ensure_metadata(self.table_metadata)

        # Load records from specified source
        df = self._load_records(db_client, context)
        self.logger.info(f"[{self.name}] Loaded {df.height} rows, {df.width} columns.")
        context.progress_update(f"Loaded {df.height} record{'s' if df.height != 1 else ''}.")

        if df.height == 0:
            self.logger.warning(f"[{self.name}] Source returned no rows.")
            context.progress_pop()
            context.progress_warning(f"No records to load into table {table}. Continuing...")
            if self.loading_strategy == LoadingStrategy.OVERWRITE:
                self.logger.warning(
                    f"[{self.name}] Overwrite will truncate {table} and leave it empty."
                )
            else:
                self.logger.info(f"[{self.name}] Nothing to load; skipping.")
                return

        # Apply transformations to the loaded data
        transform_total = len(self.transformations)
        for index, transformation in enumerate(self.transformations, start=1):
            before_rows = df.height
            transform_name = type(transformation).__name__
            context.progress_update(
                f"Transformation {index}/{transform_total}: {transform_name}..."
            )
            df = transformation.transform(df)
            self.logger.info(
                f"[{self.name}] Transformation {index}/{len(self.transformations)} "
                f"({transform_name}): {before_rows} -> {df.height} rows."
            )
            if df.height == 0:
                self.logger.info(
                    f"[{self.name}] No rows remaining after {transform_name}; "
                    "skipping remaining transformations."
                )
                break

        if df.height == 0:
            self.logger.info(f"[{self.name}] No rows after transformations; skipping save.")
            context.progress_pop()
            context.progress_success(f"No records to save to {table}.")
            return

        # Save the transformed data to the target table
        context.progress_update(
            f"Saving {df.height} record{'s' if df.height != 1 else ''} to {table}..."
        )
        result = self._save_records(db_client, self.table_metadata, df)
        self.logger.info(
            f"[{self.name}] Saved to {table}: "
            f"inserted={result.rows_inserted}, updated={result.rows_updated}, "
            f"deleted={result.rows_deleted}."
        )
        context.progress_pop()
        context.progress_success(
            f"Saved records to {table}: {result.rows_inserted} inserted, "
            f"{result.rows_updated} updated, {result.rows_deleted} deleted."
        )

    def _save_records(
        self,
        db_client: DatabaseClient,
        table_metadata: TableMetadata,
        data: pl.DataFrame,
    ) -> WriteResult:
        """Save records to the given table using the configured loading strategy."""
        db_client.ensure_metadata(table_metadata)
        try:
            table_metadata.validate_dataframe_for_load(data, log=self.logger)
        except MetadataError as e:
            self.logger.log_and_raise(e)
        self.logger.info(
            f"[{self.name}] Schema reconciled for {table_metadata.qualified_name_sql()}."
        )
        try:
            match self.loading_strategy:
                case LoadingStrategy.APPEND:
                    return db_client.insert(data, table_metadata, on_conflict="error")
                case LoadingStrategy.INSERT_IGNORE:
                    return db_client.insert(data, table_metadata, on_conflict="nothing")
                case LoadingStrategy.OVERWRITE:
                    return db_client.overwrite(
                        data,
                        table_metadata,
                        cascade=self.cascade if self.cascade is not None else False,
                    )
                case LoadingStrategy.MERGE:
                    return db_client.merge(
                        data,
                        table_metadata,
                        strategy=self.merge_strategy,
                        match_condition=self.match_condition,
                    )
                case _:
                    raise ValueError(f"Unsupported loading strategy: {self.loading_strategy.value}")
        except Exception as e:
            self.logger.log_and_raise(e)
            raise

    def _load_records(self, db_client: DatabaseClient, context: PipelineContext) -> pl.DataFrame:
        """Load records from the source into a Polars DataFrame."""
        raise NotImplementedError


class TransformStep(LoadToDatabaseStep):
    """Load data from a table, transform it and save it to another table."""

    def __init__(
        self,
        name: str,
        source_data_class: type[TableDataClass],
        target_data_class: type[TableDataClass],
        transformations: list[DataFrameTransformation] = [],
        *,
        on: str | None = None,
        source_column: str | None = None,
        loading_strategy: LoadingStrategy,
        merge_strategy: MergeStrategy | None = None,
        cascade: bool | None = None,
        match_condition: str | None = None,
        source_columns: list[str] | None = None,
    ) -> None:
        resolved_merge = merge_strategy or MergeStrategy.upsert()
        if on is not None and resolved_merge.when_not_matched_by_source == "delete":
            raise ValueError(
                "TransformStep cannot combine an incremental filter (on=...) with "
                "full_sync merge strategy; use on=None for full_reload mode."
            )

        self._incremental_filter = IncrementalFilterTransformation(
            target_data_class=target_data_class,
            on=on,
            source_column=source_column,
        )
        super().__init__(
            name=name,
            data_class=target_data_class,
            transformations=[self._incremental_filter, *transformations],
            loading_strategy=loading_strategy,
            merge_strategy=resolved_merge,
            cascade=cascade,
            match_condition=match_condition,
        )
        self.source_table_metadata = source_data_class.get_metadata()
        self.source_columns = source_columns
        self.on = on
        self.source_column = source_column if source_column is not None else on

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        scope_where = self._optional_scope_where_clause(self.table_metadata, context)
        self._incremental_filter.db_client = db_client
        self._incremental_filter.set_scope_where(scope_where)
        if (
            self.loading_strategy == LoadingStrategy.MERGE
            and self.merge_strategy.when_not_matched_by_source == "delete"
        ):
            self.match_condition = scope_where
        super().run(db_client, context)

    @staticmethod
    def _optional_scope_where_clause(
        table: TableMetadata,
        context: PipelineContext,
    ) -> str | None:
        """Return a scope WHERE clause when context and table columns allow it."""
        columns = table.column_names()
        if context.account_id is not None and "account_id" in columns:
            return generate_ident_is_literal("account_id", context.account_id)
        if context.user_id is not None and "user_id" in columns:
            return generate_ident_is_literal("user_id", context.user_id)
        return None

    @staticmethod
    def _context_where_clause(
        source_table: TableMetadata,
        context: PipelineContext,
    ) -> str | None:
        source = source_table.qualified_name_sql()
        columns = source_table.column_names()

        if context.account_id is not None:
            if "account_id" not in columns:
                raise PipelineError(
                    f"Cannot scope {source} by account_id: the source table does not "
                    f"include an account_id column. Add account_id to the source table "
                    f"metadata or run this pipeline without account_id in context."
                )
            return generate_ident_is_literal("account_id", context.account_id)

        if context.user_id is not None:
            if "user_id" not in columns:
                raise PipelineError(
                    f"Cannot scope {source} by user_id: the source table does not "
                    f"include a user_id column. Add user_id to the source table "
                    f"metadata or run this pipeline without user_id in context."
                )
            return generate_ident_is_literal("user_id", context.user_id)

        return None

    def _load_records(self, db_client: DatabaseClient, context: PipelineContext) -> pl.DataFrame:
        """Load records from the source table into a Polars DataFrame."""
        source = self.source_table_metadata.qualified_name_sql()
        if not db_client.table_exists(self.source_table_metadata):
            self.logger.warning(
                f"[{self.name}] Source table {source} does not exist; using empty frame."
            )
            return pl.DataFrame()
        context.progress_update(f"Reading records from {source}...")
        where = self._context_where_clause(self.source_table_metadata, context)
        where = self._with_incremental_anti_join(where)
        return db_client.read(
            self.source_table_metadata,
            columns=self.source_columns,
            where=where,
            as_polars=True,
        )

    def _with_incremental_anti_join(self, where: str | None) -> str | None:
        """Skip source rows whose incremental key already exists in the target table."""
        if self.on is None or self.source_column is None:
            return where

        source_columns = set(self.source_table_metadata.column_names())
        target_columns = set(self.table_metadata.column_names())
        if self.source_column not in source_columns or self.on not in target_columns:
            return where

        source_q = self.source_table_metadata.qualified_name_sql()
        target_q = self.table_metadata.qualified_name_sql()
        anti_join = (
            f"NOT EXISTS (SELECT 1 FROM {target_q} AS _inc_tgt "
            f"WHERE _inc_tgt.{quote_ident(self.on)} = "
            f"{source_q}.{quote_ident(self.source_column)}"
        )
        scope_where = self._incremental_filter.scope_where
        if scope_where:
            anti_join += f" AND ({scope_where})"
        anti_join += ")"

        if where:
            return f"({where}) AND {anti_join}"
        return anti_join


class StorageToTableStep(LoadToDatabaseStep):
    """
    Load data from object storage into a table.

    Args:
        storage_path: Key prefix to load from. Interpretation depends on ``recursive``:
            - ``recursive=False``: must be a single object key (e.g. ``data/file.jsonl``).
            - ``recursive=True``: all matching objects under the prefix are loaded.
        file_type: File format to load (also used as the required suffix, e.g. ``.jsonl``).
        quarantine_path: When set, objects that fail to load or whose batch fails to save
            are moved here, preserving relative paths under ``storage_path``.
        glob_pattern: Optional regex applied to each candidate key (POSIX form).
    """

    PRE_LOAD_TRANSFORMATIONS: ClassVar[list[DataFrameTransformation]] = [
        CastToDatetimeTransformation(columns=["_ingestion_ts"]),
    ]

    def __init__(
        self,
        name: str,
        storage_path: str,
        file_type: FileType,
        data_class: type[TableDataClass],
        transformations: list[DataFrameTransformation] = [],
        *,
        recursive: bool = True,
        glob_pattern: str | None = None,
        quarantine_path: str | None = None,
        storage: ObjectStorage | None = None,
        loading_strategy: LoadingStrategy,
        merge_strategy: MergeStrategy | None = None,
        cascade: bool | None = None,
        match_condition: str | None = None,
    ) -> None:
        super().__init__(
            name=name,
            data_class=data_class,
            transformations=self.PRE_LOAD_TRANSFORMATIONS + transformations,
            loading_strategy=loading_strategy,
            merge_strategy=merge_strategy,
            cascade=cascade,
            match_condition=match_condition,
        )
        self.storage_path = storage_path.strip("/")
        self.recursive = recursive
        self.glob_pattern = glob_pattern
        self.quarantine_path = quarantine_path.strip("/") if quarantine_path else None
        self._storage = storage
        self.file_type = file_type
        self.file_loader: FileLoader = FileLoaderFactory.get_loader(file_type, logger=self.logger)
        self._loaded_keys: list[str] = []

    def _get_storage(self) -> ObjectStorage:
        return self._storage if self._storage is not None else get_raw_storage()

    def _resolve_storage_paths(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        """
        Override when ``storage_path`` or ``quarantine_path`` depend on runtime context.

        Defaults to values set in ``__init__``. Subclasses typically set both paths
        together (e.g. ingested source + failed quarantine for the same account).
        """

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        """
        Load, transform, and save; quarantine source objects on load or save failure.

        Per-object load failures quarantine that key only. Transform/save failures
        quarantine all keys recorded in ``_loaded_keys``.
        """
        self._resolve_storage_paths(db_client, context)
        self._loaded_keys = []
        try:
            super().run(db_client, context)
        except Exception:
            self._quarantine_keys(self._loaded_keys)
            raise

    def _relative_key(self, source_key: str) -> str:
        return ObjectStorage.relative_key_under(source_key, self.storage_path)

    def _quarantine_destination_key(self, source_key: str) -> str:
        assert self.quarantine_path is not None
        relative = self._relative_key(source_key)
        destination = ObjectStorage.resolve_key(self.quarantine_path, relative)
        storage = self._get_storage()
        if storage.read_bytes(destination) is not None:
            destination = ObjectStorage.resolve_key(
                self.quarantine_path,
                ObjectStorage.unique_key_variant(relative, uuid4().hex),
            )
        return destination

    def _quarantine_keys(self, keys: list[str]) -> None:
        if self.quarantine_path is None:
            return
        storage = self._get_storage()
        for key in keys:
            destination = self._quarantine_destination_key(key)
            try:
                storage.move_verified(key, destination, overwrite=False)
                self.logger.warning(f"[{self.name}] Quarantined {key} -> {destination}.")
            except Exception as e:
                self.logger.error(f"[{self.name}] Failed to quarantine {key}: {e}")

    def _load_records(self, db_client: DatabaseClient, context: PipelineContext) -> pl.DataFrame:
        """Load records from storage into a Polars DataFrame."""
        storage = self._get_storage()
        keys = storage.list_keys(
            self.storage_path,
            recursive=self.recursive,
            suffix=self.file_type.value,
            glob_pattern=self.glob_pattern,
        )

        if not keys:
            self.logger.warning(
                f"[{self.name}] No objects found at {self.storage_path} "
                f"(recursive={self.recursive}, suffix=.{self.file_type.value}, "
                f"glob_pattern={self.glob_pattern!r})."
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
                self.logger.info(f"[{self.name}] Added metadata to {key}.")
            except Exception as e:
                self.logger.warning(f"[{self.name}] Failed to add metadata to {key}: {e}")
                self._quarantine_keys([key])
        self.logger.info(f"[{self.name}] Loaded {len(records)} records from {len(keys)} keys.")
        context.progress_update(
            f"Loaded {len(records)} record{'s' if len(records) != 1 else ''}. "
            f"Processed {len(self._loaded_keys)}/{len(keys)} file{'s' if len(self._loaded_keys) != 1 else ''} successfully."
        )

        df = pl.DataFrame(records)
        return df


class StreamToTableStep(LoadToDatabaseStep):
    """
    Load data from one or more open text streams into a table.

    Parsing is delegated to a :class:`FileLoader`; this step only iterates
    streams, collects ``list[dict]`` records, and hands them to the shared
    transform/save path on :class:`LoadToDatabaseStep`.

    Args:
        streams: Text streams to parse, each with an optional source name for
            error messages and ``_source_name`` record metadata. Override
            ``_resolve_streams`` to supply or open streams at runtime (e.g.
            after an HTTP fetch).
        file_type: File format to parse (selects the :class:`FileLoader`).
    """

    PRE_LOAD_TRANSFORMATIONS: ClassVar[list[DataFrameTransformation]] = [
        CastToDatetimeTransformation(columns=["_ingestion_ts"]),
    ]

    def __init__(
        self,
        name: str,
        file_type: FileType,
        data_class: type[TableDataClass],
        transformations: list[DataFrameTransformation] = [],
        *,
        streams: list[TextStreamSource] | None = None,
        loading_strategy: LoadingStrategy,
        merge_strategy: MergeStrategy | None = None,
        cascade: bool | None = None,
        match_condition: str | None = None,
    ) -> None:
        super().__init__(
            name=name,
            data_class=data_class,
            transformations=self.PRE_LOAD_TRANSFORMATIONS + transformations,
            loading_strategy=loading_strategy,
            merge_strategy=merge_strategy,
            cascade=cascade,
            match_condition=match_condition,
        )
        self.streams = list(streams or [])
        self.file_type = file_type
        self.file_loader: FileLoader = FileLoaderFactory.get_loader(file_type, logger=self.logger)

    def _resolve_streams(
        self, db_client: DatabaseClient, context: PipelineContext
    ) -> list[TextStreamSource]:
        """
        Override when streams depend on runtime context.

        Defaults to ``streams`` set in ``__init__``. Subclasses typically open
        or fetch streams here (e.g. ``StringIO(response.text)`` per URL).
        """
        return self.streams

    def _load_records(self, db_client: DatabaseClient, context: PipelineContext) -> pl.DataFrame:
        """Load records from text streams into a Polars DataFrame."""
        sources = self._resolve_streams(db_client, context)

        if not sources:
            self.logger.warning(f"[{self.name}] No text streams configured.")
            context.progress_pop()
            context.progress_warning("No streams to extract records from.")
            return pl.DataFrame()

        stream_total = len(sources)
        context.progress_update(
            f"Found {stream_total} stream{'s' if stream_total != 1 else ''} to load."
        )
        records: list[dict] = []
        loaded_count = 0
        for stream_index, source in enumerate(sources, start=1):
            label = source.source_name or f"stream {stream_index}"
            context.progress_update(f"Loading stream {stream_index}/{stream_total}...")
            self.logger.info(f"[{self.name}] Loading {label}.")
            try:
                file_records = self.file_loader.load_source(source)
            except Exception as e:
                self.logger.warning(f"[{self.name}] Failed to load {label}: {e}")
                continue
            self.logger.info(f"[{self.name}] Loaded {len(file_records)} records from {label}.")

            try:
                ingestion_ts = get_current_datetime()
                for record in file_records:
                    record["_source_name"] = source.source_name
                    record["_ingestion_ts"] = ingestion_ts
                records.extend(file_records)
                loaded_count += 1
                self.logger.info(f"[{self.name}] Added metadata to {label}.")
            except Exception as e:
                self.logger.warning(f"[{self.name}] Failed to add metadata to {label}: {e}")

        self.logger.info(
            f"[{self.name}] Loaded {len(records)} records from {stream_total} stream(s)."
        )
        context.progress_update(
            f"Loaded {len(records)} record{'s' if len(records) != 1 else ''}. "
            f"Processed {loaded_count}/{stream_total} stream{'s' if stream_total != 1 else ''} successfully."
        )

        return pl.DataFrame(records)
