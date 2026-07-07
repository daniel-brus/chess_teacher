# This file is deliberately extremely small: It is a thin base module to construct DataFrame transformations from.
from abc import ABC, abstractmethod

import polars as pl


class DataFrameTransformation(ABC):
    """Base class for all DataFrame transformations."""

    @abstractmethod
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform the DataFrame."""
        pass
