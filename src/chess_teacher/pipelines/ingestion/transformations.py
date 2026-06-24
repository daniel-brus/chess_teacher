from __future__ import annotations

import json
import re
from datetime import date
from pathlib import PurePosixPath
from typing import Any

import polars as pl

from chess_teacher.platform.account import AccountPlatform
from chess_teacher.utils.exception_utils import DataError, TransformationError
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.pipeline_utils.transformations import DataFrameTransformation

logger = get_logger()

_PIPELINE_ONLY_COLUMNS = frozenset({"_source_file", "_ingestion_ts"})


def is_chess_com_expr() -> pl.Expr:
    """True when the row is from Chess.com."""
    return pl.col("platform") == AccountPlatform.CHESS_COM.value


def is_lichess_expr() -> pl.Expr:
    """True when the row is from Lichess."""
    return pl.col("platform") == AccountPlatform.LICHESS.value


def chain_when(branches: list[tuple[pl.Expr, pl.Expr]], *, default: pl.Expr) -> pl.Expr:
    """Fold ``(condition, value)`` pairs into a nested ``pl.when`` chain."""
    expr = default
    for condition, value in reversed(branches):
        expr = pl.when(condition).then(value).otherwise(expr)
    return expr


class SerializeRawResponseTransformation(DataFrameTransformation):
    """
    Snapshot each loaded platform record as ``raw_response`` JSON before joins.

    Persists ``source_file`` and ``ingested_at`` from pipeline-only load columns.
    """

    SOURCE_FILE_COLUMN = "_source_file"
    INGESTION_TS_COLUMN = "_ingestion_ts"
    RAW_RESPONSE_COLUMN = "raw_response"
    SOURCE_FILE_OUT = "source_file"
    INGESTED_AT_COLUMN = "ingested_at"

    _OUTPUT_DTYPE = pl.Struct({
        RAW_RESPONSE_COLUMN: pl.Utf8,
        SOURCE_FILE_OUT: pl.Utf8,
        INGESTED_AT_COLUMN: pl.Datetime(time_zone="UTC"),
    })

    @classmethod
    def _serialize_row(cls, row: dict[str, Any]) -> dict[str, Any]:
        payload = {key: value for key, value in row.items() if key not in _PIPELINE_ONLY_COLUMNS}
        return {
            cls.RAW_RESPONSE_COLUMN: json.dumps(payload, default=str),
            cls.SOURCE_FILE_OUT: row.get(cls.SOURCE_FILE_COLUMN),
            cls.INGESTED_AT_COLUMN: row.get(cls.INGESTION_TS_COLUMN),
        }

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        for column in (self.SOURCE_FILE_COLUMN, self.INGESTION_TS_COLUMN):
            if column not in df.columns:
                logger.log_and_raise(
                    TransformationError(f"Column {column!r} is required to serialize raw_response.")
                )

        try:
            return df.with_columns(
                pl
                .struct([pl.col(column) for column in df.columns])
                .map_elements(self._serialize_row, return_dtype=self._OUTPUT_DTYPE)
                .alias("_raw_snapshot")
            ).unnest("_raw_snapshot")
        except Exception as e:
            logger.log_and_raise(TransformationError(f"Failed to serialize raw_response: {e}"))


class ExtractFileMetadataTransformation(DataFrameTransformation):
    """
    Extract ingestion file metadata from ``_source_file`` paths.

    Expected layout:
        .../ingested/{account_id}/{YYYY}/{MM}/{DD}/{platform}_{batch_id}.jsonl
    """

    SOURCE_FILE_COLUMN = "_source_file"
    INGESTED_FOLDER = "ingested"
    _SOURCE_FILE_PATH_RE = re.compile(
        rf"(?:^|.*/){re.escape(INGESTED_FOLDER)}"
        r"/(?P<account_id>[^/]+)/(?P<year>\d{4})/(?P<month>\d{2})/(?P<day>\d{2})/(?P<file_name>[^/]+)$"
    )

    @staticmethod
    def _empty_metadata() -> dict[str, str | date | None]:
        return {
            "account_id": None,
            "ingestion_date": None,
            "file_name": None,
        }

    @classmethod
    def _parse_source_file_path(cls, source_file: str) -> dict[str, str | date | None]:
        """Parse account_id, ingestion_date, file_name from a _source_file path."""
        if not source_file:
            return cls._empty_metadata()

        normalized = source_file.replace("\\", "/")
        match = cls._SOURCE_FILE_PATH_RE.search(normalized)
        if match is None:
            return cls._parse_source_file_path_fallback(normalized)

        file_name = match.group("file_name")
        return {
            "account_id": match.group("account_id"),
            "ingestion_date": date(
                int(match.group("year")),
                int(match.group("month")),
                int(match.group("day")),
            ),
            "file_name": file_name,
        }

    @classmethod
    def _parse_source_file_path_fallback(cls, normalized: str) -> dict[str, str | date | None]:
        """Fallback parser using path parts when the regex does not match."""
        parts = PurePosixPath(normalized).parts
        try:
            ingested_idx = parts.index(cls.INGESTED_FOLDER)
        except ValueError:
            return cls._empty_metadata()

        tail = parts[ingested_idx + 1 :]
        if len(tail) < 5:
            file_name = tail[-1] if tail else None
            return {
                "account_id": tail[0] if tail else None,
                "ingestion_date": None,
                "file_name": file_name,
            }

        account_id, year, month, day, file_name = tail[0], tail[1], tail[2], tail[3], tail[4]
        ingestion_date: date | None = None
        if len(year) == 4 and year.isdigit() and month.isdigit() and day.isdigit():
            ingestion_date = date(int(year), int(month), int(day))

        return {
            "account_id": account_id,
            "ingestion_date": ingestion_date,
            "file_name": file_name,
        }

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.SOURCE_FILE_COLUMN not in df.columns:
            logger.log_and_raise(
                TransformationError(
                    f"Column {self.SOURCE_FILE_COLUMN!r} is required for file metadata extraction."
                )
            )

        try:
            result = df.with_columns(
                pl
                .col(self.SOURCE_FILE_COLUMN)
                .map_elements(
                    self._parse_source_file_path,
                    return_dtype=pl.Struct({
                        "account_id": pl.Utf8,
                        "ingestion_date": pl.Date,
                        "file_name": pl.Utf8,
                    }),
                )
                .alias("_file_metadata")
            ).unnest("_file_metadata")
        except Exception as e:
            logger.log_and_raise(TransformationError(f"Failed to extract file metadata: {e}"))

        unparsed = result.filter(pl.col("account_id").is_null()).height
        if unparsed:
            logger.warning(
                "ExtractFileMetadataTransformation: %s row(s) could not be parsed from %s.",
                unparsed,
                self.SOURCE_FILE_COLUMN,
            )

        return result


class ExtractPlatformGameIdTransformation(DataFrameTransformation):
    """
    Extract the platform game ID from the loaded record.
    Requires:
    - the input DataFrame to contain the 'platform' column.
    Returns the input DataFrame with only these columns added (or updated if already present):
    - platform_game_id (str: the game ID on the platform)
    """

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """If the platform is Chess.com, the platform game ID is the "uuid" field in the record.
        If the platform is Lichess, the platform game ID is the "id" field in the record.
        """
        if "platform" not in df.columns:
            logger.log_and_raise(
                TransformationError(
                    "Column 'platform' is required for platform game ID extraction."
                )
            )
        column_names = set(df.columns)
        branches: list[tuple[pl.Expr, pl.Expr]] = []
        if "uuid" in column_names:
            branches.append((is_chess_com_expr(), pl.col("uuid")))
        if "id" in column_names:
            branches.append((is_lichess_expr(), pl.col("id")))

        try:
            df = df.with_columns(
                platform_game_id=chain_when(branches, default=pl.lit(None).cast(pl.Utf8))
            )
        except Exception as e:
            logger.log_and_raise(TransformationError(f"Failed to extract platform game ID: {e}"))

        failed_rows = df.filter(pl.col("platform_game_id").is_null()).height
        if failed_rows:
            logger.log_and_raise(
                DataError(f"Failed to extract platform game ID for {failed_rows} rows.")
            )
        return df
