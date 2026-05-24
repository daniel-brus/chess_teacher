from abc import ABC, abstractmethod
from typing import Literal

import polars as pl

from chess_teacher.utils.db_client import DatabaseClient, get_db_client
from chess_teacher.utils.exception_utils import TransformationError
from chess_teacher.utils.logging_utils import get_logger
from chess_teacher.utils.metadata_utils import TableMetadata

logger = get_logger()


class DataFrameTransformation(ABC):
    """Base class for all DataFrame transformations."""

    @abstractmethod
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform the DataFrame."""
        pass


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
        with_table_metadata: TableMetadata,
        how: Literal["inner", "left", "right", "full", "semi", "anti", "cross"] = "left",
        left_on: list[str] | None = None,
        right_on: list[str] | None = None,
        where: str | None = None,
        db_client: DatabaseClient | None = None,
    ):
        """
        Join a DataFrame with a table.
        Args:
            with_table_metadata: TableMetadata of the table to join with
            how: How to join the tables (inner, left, right, outer)
            left_on: Columns to join on the left table (default: primary key of the table to join with)
            right_on: Columns to join on the right table (default: primary key of the table to join with)
            where: Optional where clause to filter the other table before joining
            db_client: DatabaseClient to use (default: get_db_client())
        """
        super().__init__()
        self.with_table_metadata = with_table_metadata
        self.db_client = db_client or get_db_client()
        self.how = how
        self.left_on = left_on or with_table_metadata.primary_key
        self.right_on = right_on or with_table_metadata.primary_key
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
