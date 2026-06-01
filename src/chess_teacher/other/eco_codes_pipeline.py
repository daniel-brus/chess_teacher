from __future__ import annotations

from io import StringIO

import requests

from chess_teacher.other.dataclasses import RawEcoCode
from chess_teacher.pipelines.pipeline_base import Pipeline, PipelineContext
from chess_teacher.pipelines.pipeline_helpers import PipelineRunResult
from chess_teacher.pipelines.pipeline_steps import (
    LoadingStrategy,
    StreamToTableStep,
)
from chess_teacher.pipelines.transformations import (
    AssertUniqueColumnsTransformation,
    CreateHashedIdTransformation,
    RenameColumnsTransformation,
)
from chess_teacher.utils.db_client import DatabaseClient
from chess_teacher.utils.file_loader import TextStreamSource
from chess_teacher.utils.file_utils import FileType


class LoadLichessEcoCodesStep(StreamToTableStep):
    """Fetch lichess chess-openings TSV files and load them into raw_eco_codes."""

    _LICHESS_OPENINGS_BASE_URL = (
        "https://raw.githubusercontent.com/lichess-org/chess-openings/master"
    )
    _LICHESS_ECO_FILES = ("a", "b", "c", "d", "e")

    def __init__(self) -> None:
        super().__init__(
            name="LoadLichessEcoCodes",
            file_type=FileType.TSV,
            data_class=RawEcoCode,
            transformations=[
                RenameColumnsTransformation({"eco": "eco_code"}),
                AssertUniqueColumnsTransformation("pgn", label="PGN"),
                CreateHashedIdTransformation(data_class=RawEcoCode),
            ],
            loading_strategy=LoadingStrategy.OVERWRITE,
        )

    def _resolve_streams(
        self, db_client: DatabaseClient, context: PipelineContext
    ) -> list[TextStreamSource]:
        sources: list[TextStreamSource] = []
        for letter in self._LICHESS_ECO_FILES:
            url = f"{self._LICHESS_OPENINGS_BASE_URL}/{letter}.tsv"
            self.logger.info(f"[{self.name}] Fetching {url}.")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            sources.append(TextStreamSource(StringIO(response.text), source_name=url))
        return sources


def run_load_lichess_eco_codes_pipeline() -> PipelineRunResult:
    """Fetch lichess ECO TSV files and overwrite other.raw_eco_codes."""
    pipeline = Pipeline(
        name="load_lichess_eco_codes",
        steps=[LoadLichessEcoCodesStep()],
    )
    return pipeline.run()
