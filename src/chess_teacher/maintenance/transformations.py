"""DataFrame transformations for log ingestion pipeline steps."""

from __future__ import annotations

from typing import cast

import polars as pl

from chess_teacher.utils.exception_utils import TransformationError
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.pipeline_utils.dataframe_transformation import DataFrameTransformation

logger = get_logger()

WARNING_ERROR_LEVELS: frozenset[str] = frozenset({"WARNING", "ERROR", "CRITICAL"})
NO_EXC_TYPE_LABEL = "(no exception)"

_LOG_LEVEL_HOURLY_GROUP_COLUMNS = (
    "bucket_start",
    "environment",
    "level",
    "logger",
    "hostname",
)
_EXCEPTION_HOURLY_GROUP_COLUMNS = (
    "bucket_start",
    "environment",
    "level",
    "exc_type",
)

_EMPTY_LOG_LEVEL_HOURLY_SCHEMA = pl.Schema(
    cast(
        dict[str, pl.DataType],
        {
            "bucket_start": pl.Datetime(time_zone="UTC"),
            "environment": pl.Utf8,
            "level": pl.Utf8,
            "logger": pl.Utf8,
            "hostname": pl.Utf8,
            "log_count": pl.Int64,
        },
    )
)
_EMPTY_EXCEPTION_HOURLY_SCHEMA = pl.Schema(
    cast(
        dict[str, pl.DataType],
        {
            "bucket_start": pl.Datetime(time_zone="UTC"),
            "environment": pl.Utf8,
            "level": pl.Utf8,
            "exc_type": pl.Utf8,
            "exception_count": pl.Int64,
        },
    )
)


class ParseLogTimestampColumnsTransformation(DataFrameTransformation):
    """Parse ISO timestamp strings from JSON log records into UTC datetimes."""

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.height == 0 or "ts" not in df.columns:
            return df
        if df.schema["ts"] != pl.Utf8:
            return df
        try:
            return df.with_columns(pl.col("ts").str.to_datetime(time_zone="UTC").alias("ts"))
        except Exception as e:
            logger.log_and_raise(TransformationError(f"Failed to parse log ts column: {e}"))


class FilterWarningErrorLevelsTransformation(DataFrameTransformation):
    """Keep only WARNING, ERROR, and CRITICAL log rows."""

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.height == 0:
            return df
        if "level" not in df.columns:
            logger.log_and_raise(
                TransformationError("Column 'level' is required to filter warning/error logs.")
            )
        return df.filter(pl.col("level").is_in(list(WARNING_ERROR_LEVELS)))


class AggregateLogLevelHourlyTransformation(DataFrameTransformation):
    """Roll up raw log rows into hourly counts by environment, level, logger, and host."""

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.height == 0:
            return pl.DataFrame(schema=_EMPTY_LOG_LEVEL_HOURLY_SCHEMA)

        required = {"ts", "environment", "level", "logger", "source_file"}
        missing = required - set(df.columns)
        if missing:
            logger.log_and_raise(
                TransformationError(
                    f"Missing columns for log level hourly aggregation: {sorted(missing)}"
                )
            )

        try:
            from chess_teacher.maintenance.pipeline_steps import LoadRawLogsStep

            prepared = df.with_columns(
                pl.col("ts").dt.truncate("1h").alias("bucket_start"),
                pl
                .col("source_file")
                .map_elements(LoadRawLogsStep.parse_closed_log_hostname, return_dtype=pl.Utf8)
                .alias("hostname"),
            )
            return (
                prepared
                .group_by(list(_LOG_LEVEL_HOURLY_GROUP_COLUMNS))
                .len()
                .rename({"len": "log_count"})
                .sort(_LOG_LEVEL_HOURLY_GROUP_COLUMNS)
            )
        except Exception as e:
            logger.log_and_raise(
                TransformationError(f"Failed to aggregate log level hourly counts: {e}")
            )


class AggregateExceptionHourlyTransformation(DataFrameTransformation):
    """Roll up warning/error log rows into hourly counts by level and exception type."""

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.height == 0:
            return pl.DataFrame(schema=_EMPTY_EXCEPTION_HOURLY_SCHEMA)

        required = {"ts", "environment", "level"}
        missing = required - set(df.columns)
        if missing:
            logger.log_and_raise(
                TransformationError(
                    f"Missing columns for exception hourly aggregation: {sorted(missing)}"
                )
            )

        try:
            prepared = df.filter(pl.col("level").is_in(list(WARNING_ERROR_LEVELS))).with_columns(
                pl.col("ts").dt.truncate("1h").alias("bucket_start"),
                pl
                .when(pl.col("exc_type").is_not_null())
                .then(pl.col("exc_type"))
                .otherwise(pl.lit(NO_EXC_TYPE_LABEL))
                .alias("exc_type")
                if "exc_type" in df.columns
                else pl.lit(NO_EXC_TYPE_LABEL).alias("exc_type"),
            )
            if prepared.height == 0:
                return pl.DataFrame(schema=_EMPTY_EXCEPTION_HOURLY_SCHEMA)
            return (
                prepared
                .group_by(list(_EXCEPTION_HOURLY_GROUP_COLUMNS))
                .len()
                .rename({"len": "exception_count"})
                .sort(_EXCEPTION_HOURLY_GROUP_COLUMNS)
            )
        except Exception as e:
            logger.log_and_raise(
                TransformationError(f"Failed to aggregate exception hourly counts: {e}")
            )
