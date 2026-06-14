from __future__ import annotations

from io import StringIO

import requests

from chess_teacher.other.chess_com_openings import refresh_slug_title_lookup_from_database
from chess_teacher.other.dataclasses import RawEcoCode
from chess_teacher.pipelines.pipeline_base import Pipeline, PipelineContext, PipelineStep
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
from chess_teacher.utils.db.client import DatabaseClient
from chess_teacher.utils.files.file_utils import FileType, TextStreamSource


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


class RefreshChessComOpeningSlugsStep(PipelineStep):
    """Scan ``raw_games`` for slugs and fetch missing titles into ``other.raw_chess_com_openings``."""

    def __init__(self, *, request_delay_s: float = 0.25) -> None:
        super().__init__(name="RefreshChessComOpeningSlugs")
        self.request_delay_s = request_delay_s

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        result = refresh_slug_title_lookup_from_database(
            db_client,
            request_delay_s=self.request_delay_s,
            logger=self.logger,
        )
        context.progress_success(
            f"Chess.com openings: {result.fetched} fetched, "
            f"{result.unresolved} unresolved, "
            f"{result.distinct_slugs} distinct slug(s)."
        )


def run_update_opening_lookup_tables_pipeline() -> PipelineRunResult:
    """Refresh ``other.raw_eco_codes`` and ``other.raw_chess_com_openings``."""
    pipeline = Pipeline(
        name="update_opening_lookup_tables",
        steps=[
            LoadLichessEcoCodesStep(),
            RefreshChessComOpeningSlugsStep(),
        ],
    )
    return pipeline.run()
