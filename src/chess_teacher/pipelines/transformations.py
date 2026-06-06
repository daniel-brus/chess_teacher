from abc import ABC, abstractmethod
from typing import Literal

import polars as pl

from chess_teacher.utils.db_client import DatabaseClient, get_db_client
from chess_teacher.utils.exception_utils import ConfigError, TransformationError
from chess_teacher.utils.logging_utils import get_logger
from chess_teacher.utils.metadata_utils import ColumnMetadata
from chess_teacher.utils.table_data_class import TableDataClass

logger = get_logger()


class DataFrameTransformation(ABC):
    """Base class for all DataFrame transformations."""

    @abstractmethod
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform the DataFrame."""
        pass


class CreateHashedIdTransformation(DataFrameTransformation):
    """Create hashed primary key column from TableDataClass id-hash columns."""

    def __init__(
        self,
        data_class: type[TableDataClass],
        *,
        target_column: str | None = None,
    ):
        super().__init__()
        self.data_class = data_class
        self.metadata = data_class.get_metadata()
        self.id_hash_columns = data_class.get_id_hash_columns()
        if target_column is None:
            pk_cols = self.metadata.primary_key
            if len(pk_cols) != 1:
                logger.log_and_raise(
                    TransformationError(
                        f"target_column required when primary_key has {len(pk_cols)} "
                        f"columns: {pk_cols}"
                    )
                )
            self.target_column = pk_cols[0]
        else:
            self.target_column = target_column

    def _generate_id_from_row(self, row: dict[str, object]) -> str:
        try:
            return self.data_class.generate_id(row)
        except ConfigError as exc:
            logger.log_and_raise(TransformationError(str(exc)))
            raise

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.id_hash_columns:
            logger.log_and_raise(
                TransformationError(
                    f"{self.data_class.__name__}.get_id_hash_columns() returned no columns."
                )
            )

        df_columns = set(df.columns)
        missing = [col for col in self.id_hash_columns if col not in df_columns]
        if missing:
            logger.log_and_raise(
                TransformationError(f"Missing id hash columns in DataFrame: {missing}")
            )

        try:
            df = df.with_columns(
                pl
                .struct([pl.col(col) for col in self.id_hash_columns])
                .map_elements(
                    self._generate_id_from_row,
                    return_dtype=pl.Utf8,
                )
                .alias(self.target_column)
            )
        except TransformationError:
            raise
        except Exception as e:
            logger.log_and_raise(TransformationError(f"Failed to create hashed ID: {e}"))
        return df


class CastDataTypeTransformation(DataFrameTransformation):
    """
    Cast metadata columns present in the DataFrame to their Polars dtypes.
    Leaves other columns unchanged. Warns and skips metadata columns missing from the frame.

    Nullable columns and columns with a DB default use non-strict cast; values that cannot
    convert become null, then defaults are applied where configured. Required columns without
    a default use strict cast and fail on invalid values.
    """

    def __init__(self, data_class: type[TableDataClass]):
        super().__init__()
        self.data_class = data_class
        self.metadata = data_class.get_metadata()

    @staticmethod
    def _cast_column_expr(col: ColumnMetadata) -> pl.Expr:
        dtype = col.polars_dtype()
        source = pl.col(col.name)
        if col.nullable or col.default is not None:
            casted = source.cast(dtype, strict=False)
            if col.default is not None:
                return casted.fill_null(pl.lit(col.default).cast(dtype, strict=False))
            return casted
        return source.cast(dtype)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        try:
            df_columns = set(df.columns)
            casts: list[pl.Expr] = []
            for col in self.metadata.columns:
                if col.name not in df_columns:
                    logger.warning(
                        "Column %s from metadata not in DataFrame; skipping cast.",
                        col.name,
                    )
                    continue
                casts.append(self._cast_column_expr(col).alias(col.name))
            if casts:
                df = df.with_columns(casts)
        except Exception as e:
            logger.log_and_raise(TransformationError(f"Failed to cast data types: {e}"))
        return df


class FilterColumnsTransformation(DataFrameTransformation):
    """
    Keep metadata columns present in the DataFrame (metadata column order).
    Errors on missing required columns; warns and skips missing optional columns.
    """

    def __init__(self, data_class: type[TableDataClass]):
        super().__init__()
        self.data_class = data_class
        self.metadata = data_class.get_metadata()

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        try:
            df_columns = set(df.columns)
            required = set(self.metadata.required_load_columns())
            selected: list[str] = []
            for col in self.metadata.columns:
                if col.name not in df_columns:
                    if col.name in required:
                        logger.log_and_raise(
                            TransformationError(
                                f"Required column {col.name} not found in DataFrame."
                            )
                        )
                    logger.warning(
                        "Optional column %s from metadata not in DataFrame; skipping.",
                        col.name,
                    )
                    continue
                selected.append(col.name)
            df = df.select(selected)
        except TransformationError:
            raise
        except Exception as e:
            logger.log_and_raise(TransformationError(f"Failed to filter columns: {e}"))
        return df


class CastToDatetimeTransformation(DataFrameTransformation):
    """Cast specified columns to a datetime."""

    def __init__(self, columns: str | list[str], time_zone: str = "UTC"):
        """
        Cast given columns to a datetime.
        Args:
            columns: The columns to cast to a datetime
            time_zone: The time zone to cast the column to (default: UTC)
        """
        super().__init__()
        self.columns = columns if isinstance(columns, list) else [columns]
        self.time_zone = time_zone

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        try:
            result = df.with_columns(
                pl.col(self.columns).cast(pl.Datetime(time_zone=self.time_zone))
            )
        except Exception as e:
            logger.log_and_raise(TransformationError(f"Failed to cast columns to datetime: {e}"))
        return result


class JoinWithTableTransformation(DataFrameTransformation):
    """Join a DataFrame with a table."""

    def __init__(
        self,
        with_data_class: type[TableDataClass],
        how: Literal["inner", "left", "right", "full", "semi", "anti", "cross"] = "left",
        left_on: list[str] | None = None,
        right_on: list[str] | None = None,
        where: str | None = None,
        db_client: DatabaseClient | None = None,
    ):
        """
        Join a DataFrame with a table.
        Args:
            with_data_class: TableDataClass of the table to join with
            how: How to join the tables (inner, left, right, outer)
            left_on: Columns to join on the left table (default: primary key of the table to join with)
            right_on: Columns to join on the right table (default: primary key of the table to join with)
            where: Optional where clause to filter the other table before joining
            db_client: DatabaseClient to use (default: get_db_client())
        """
        super().__init__()
        self.with_data_class = with_data_class
        self.with_table_metadata = with_data_class.get_metadata()
        self.db_client = db_client or get_db_client()
        self.how = how
        self.left_on = left_on or self.with_table_metadata.primary_key
        self.right_on = right_on or self.with_table_metadata.primary_key
        self.where = where

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        try:
            df_other = self.db_client.read(
                self.with_table_metadata, as_polars=True, where=self.where
            )
            result = df.join(df_other, left_on=self.left_on, right_on=self.right_on, how=self.how)
        except Exception as e:
            logger.log_and_raise(TransformationError(f"Failed to join with table: {e}"))
        return result


class RenameColumnsTransformation(DataFrameTransformation):
    """Rename DataFrame columns.
    Throw an error if a column is missing, or if a column is renamed to a column that already exists."""

    def __init__(self, mapping: dict[str, str]):
        super().__init__()
        self.mapping = mapping

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        missing = [source for source in self.mapping if source not in df.columns]
        if missing:
            logger.log_and_raise(TransformationError(f"Cannot rename missing columns: {missing}"))

        target_already_existing = [
            target for target in self.mapping.values() if target in df.columns
        ]
        if target_already_existing:
            logger.log_and_raise(
                TransformationError(
                    f"Cannot rename to columns that already exist: {target_already_existing}"
                )
            )
        try:
            return df.rename(self.mapping)
        except Exception as e:
            logger.log_and_raise(TransformationError(f"Failed to rename columns: {e}"))


class AssertUniqueColumnsTransformation(DataFrameTransformation):
    """Fail when the combination of given columns are not unique across all rows."""

    def __init__(self, columns: str | list[str], *, label: str | None = None):
        super().__init__()
        self.columns = columns if isinstance(columns, list) else [columns]
        self.label = label or ", ".join(self.columns)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        missing = [column for column in self.columns if column not in df.columns]
        if missing:
            logger.log_and_raise(
                TransformationError(f"Missing columns for uniqueness check: {missing}")
            )

        unique_count = df.select(self.columns).unique().height
        if unique_count == df.height:
            logger.info(
                "Uniqueness check passed for %s (%s rows).",
                self.label,
                df.height,
            )
            return df

        duplicate_count = df.height - unique_count
        sample = (
            df
            .group_by(self.columns)
            .len()
            .filter(pl.col("len") > 1)
            .sort("len", descending=True)
            .head(5)
        )
        logger.log_and_raise(
            TransformationError(
                f"{self.label} is not unique: {duplicate_count} duplicate row(s). "
                f"Sample duplicate groups: {sample.to_dicts()}"
            )
        )
