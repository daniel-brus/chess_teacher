# This file is deliberately extremely small: It is a thin base module to construct DataFrame transformations from.
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from chess_teacher.utils.db.client import DatabaseClient
    from chess_teacher.utils.metadata_utils import TableMetadata


class DataFrameTransformation(ABC):
    """Base class for all DataFrame transformations."""

    def bind_checkpoint(
        self,
        *,
        db_client: DatabaseClient,
        table_metadata: TableMetadata,
    ) -> None:
        """Optional hook to persist partial column values during expensive transforms."""

    @abstractmethod
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform the DataFrame."""
