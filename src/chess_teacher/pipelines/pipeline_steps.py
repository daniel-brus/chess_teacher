from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import ClassVar
from uuid import uuid4

import polars as pl

from chess_teacher.pipelines.pipeline_base import PipelineContext, PipelineStep
from chess_teacher.pipelines.transformations import (
    CastDataTypeTransformation,
    CastToDatetimeTransformation,
    DataFrameTransformation,
    FilterColumnsTransformation,
)
from chess_teacher.utils.db_client import DatabaseClient, MergeStrategy, WriteResult
from chess_teacher.utils.exception_utils import MetadataError
from chess_teacher.utils.file_loader import FileLoader, FileLoaderFactory, TextStreamSource
from chess_teacher.utils.file_utils import FileType, discover_files, move_file
from chess_teacher.utils.general_utils import get_current_datetime
from chess_teacher.utils.metadata_utils import TableMetadata
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
        default_transformations = [
            transformation(data_class) for transformation in self.DEFAULT_TRANSFORMATIONS
        ]
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
        db_client.ensure_table(table_metadata)
        try:
            table_metadata.validate_dataframe_for_load(data, log=self.logger)
        except MetadataError as e:
            self.logger.log_and_raise(e)
        self.logger.info(
            f"[{self.name}] Schema check OK for {table_metadata.qualified_name_sql()}."
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
        loading_strategy: LoadingStrategy,
        merge_strategy: MergeStrategy | None = None,
        cascade: bool | None = None,
        match_condition: str | None = None,
    ) -> None:
        super().__init__(
            name=name,
            data_class=target_data_class,
            transformations=transformations,
            loading_strategy=loading_strategy,
            merge_strategy=merge_strategy,
            cascade=cascade,
            match_condition=match_condition,
        )
        self.source_table_metadata = source_data_class.get_metadata()

    def _load_records(self, db_client: DatabaseClient, context: PipelineContext) -> pl.DataFrame:
        """Load records from the source table into a Polars DataFrame."""
        source = self.source_table_metadata.qualified_name_sql()
        context.progress_update(f"Reading records from {source}...")
        return db_client.read(self.source_table_metadata, as_polars=True)


class StorageToTableStep(LoadToDatabaseStep):
    """
    Load data from storage into a table.

    Args:
        storage_path: Path to load from. Interpretation depends on ``recursive``:
            - ``recursive=False``: must be a single file (e.g. ``data/file.jsonl``).
            - ``recursive=True``: must be a directory; all matching files under it
              (including subdirectories) are loaded and concatenated.
        file_type: File format to load (also used as the required suffix, e.g. ``.jsonl``).
        quarantine_path: When set, files that fail to load or whose batch fails to save
            are moved here, preserving relative paths under ``storage_path`` (same layout
            as archive). Successfully saved files are left in place (use a follow-up
            archive step to move them to backup storage).
        glob_pattern: Optional regex applied to each candidate path (POSIX form).
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
        quarantine_path: str | Path | None = None,
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
        self.storage_path = storage_path
        self.recursive = recursive
        self.glob_pattern = glob_pattern
        self.quarantine_path = Path(quarantine_path) if quarantine_path is not None else None
        self.file_type = file_type
        self.file_loader: FileLoader = FileLoaderFactory.get_loader(file_type, logger=self.logger)
        self._loaded_paths: list[Path] = []

    def _resolve_storage_paths(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        """
        Override when ``storage_path`` or ``quarantine_path`` depend on runtime context.

        Defaults to values set in ``__init__``. Subclasses typically set both paths
        together (e.g. ingested source + failed quarantine for the same account).
        """

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        """
        Load, transform, and save; quarantine source files on load or save failure.

        Per-file load failures quarantine that file only. Transform/save failures
        quarantine all paths recorded in ``_loaded_paths``.
        """
        self._resolve_storage_paths(db_client, context)
        self._loaded_paths = []
        try:
            super().run(db_client, context)
        except Exception:
            self._quarantine_paths(self._loaded_paths)
            raise

    def _quarantine_destination(self, source: Path) -> Path:
        assert self.quarantine_path is not None
        storage_root = Path(self.storage_path)
        if self.recursive:
            relative = source.relative_to(storage_root)
        else:
            relative = Path(source.name)
        destination = self.quarantine_path / relative
        if not destination.exists():
            return destination
        return destination.with_name(f"{destination.stem}_{uuid4().hex}{destination.suffix}")

    def _quarantine_paths(self, paths: list[Path]) -> None:
        if self.quarantine_path is None:
            return
        for path in paths:
            if not path.exists():
                continue
            destination = self._quarantine_destination(path)
            try:
                move_file(
                    path,
                    destination,
                    overwrite=False,
                    mkdir=True,
                    logger=self.logger,
                )
                self.logger.warning(f"[{self.name}] Quarantined {path} -> {destination}.")
            except Exception as e:
                self.logger.error(f"[{self.name}] Failed to quarantine {path}: {e}")

    def _load_records(self, db_client: DatabaseClient, context: PipelineContext) -> pl.DataFrame:
        """Load records from storage into a Polars DataFrame."""
        paths = discover_files(
            Path(self.storage_path),
            recursive=self.recursive,
            suffix=self.file_type.value,
            glob_pattern=self.glob_pattern,
            logger=self.logger,
        )

        if not paths:
            self.logger.warning(
                f"[{self.name}] No files found at {self.storage_path} "
                f"(recursive={self.recursive}, suffix=.{self.file_type.value}, "
                f"glob_pattern={self.glob_pattern!r})."
            )
            context.progress_pop()
            context.progress_warning("No files found to extract records from.")
            return pl.DataFrame()

        file_total = len(paths)
        context.progress_update(
            f"Found {file_total} file {'s' if file_total != 1 else ''} to load."
        )
        records: list[dict] = []
        for file_index, path in enumerate(paths, start=1):
            context.progress_update(f"Loading file {file_index}/{file_total}...")
            self.logger.info(f"[{self.name}] Loading {path}.")
            try:
                with path.open(encoding="utf-8-sig") as stream:
                    source = TextStreamSource(stream, source_name=path.as_posix())
                    file_records = self.file_loader.load_source(source)
            except Exception as e:
                self.logger.warning(f"[{self.name}] Failed to load {path}: {e}")
                self._quarantine_paths([path])
                continue
            self.logger.info(f"[{self.name}] Loaded {len(file_records)} records from {path}.")

            # add filename to records as metadata
            try:
                source_file = path.resolve().as_posix()
                ingestion_ts = get_current_datetime()
                for record in file_records:
                    record["_source_file"] = source_file
                    record["_ingestion_ts"] = ingestion_ts
                records.extend(file_records)
                self._loaded_paths.append(path)
                self.logger.info(f"[{self.name}] Added metadata to {path}.")
            except Exception as e:
                self.logger.warning(f"[{self.name}] Failed to add metadata to {path}: {e}")
                self._quarantine_paths([path])
        self.logger.info(f"[{self.name}] Loaded {len(records)} records from {len(paths)} paths.")
        context.progress_update(
            f"Loaded {len(records)} record{'s' if len(records) != 1 else ''}. "
            f"Processed {len(self._loaded_paths)}/{len(paths)} file{'s' if len(self._loaded_paths) != 1 else ''} successfully."
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
