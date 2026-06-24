from chess_teacher.other.dataclasses import RawEcoCode
from chess_teacher.pipelines.ingestion.raw_games import RawGame
from chess_teacher.pipelines.modes import PipelineMode, preprocessing_transform_config
from chess_teacher.pipelines.preprocessing.games import Game
from chess_teacher.pipelines.preprocessing.move_extraction import ExtractUserMovesTransformation
from chess_teacher.pipelines.preprocessing.moves import Move
from chess_teacher.pipelines.preprocessing.transformations import (
    ApplyChessComOpeningLookupTransformation,
    ApplyLichessOpeningNameTransformation,
    CleanPGNTransformation,
    DeriveOpeningTransformation,
    ExpandRawResponseTransformation,
    ExtractGameMetadataTransformation,
    ExtractPlayersAndResultTransformation,
    FilterGamesWithPGNTransformation,
)
from chess_teacher.platform.account import Account
from chess_teacher.utils.pipeline_utils.pipeline_steps import (
    LoadingStrategy,
    TransformStep,
)
from chess_teacher.utils.pipeline_utils.transformations import (
    CreateHashedIdTransformation,
    JoinWithTableTransformation,
    RenameColumnsTransformation,
)


def _raw_games_to_games_transformations() -> list:
    return [
        ExpandRawResponseTransformation(),
        JoinWithTableTransformation(with_data_class=Account),
        FilterGamesWithPGNTransformation(),
        RenameColumnsTransformation({"pgn": "raw_pgn"}),
        ExtractGameMetadataTransformation(),
        ApplyLichessOpeningNameTransformation(),
        ExtractPlayersAndResultTransformation(),
        CleanPGNTransformation(),
        JoinWithTableTransformation(
            with_data_class=RawEcoCode,
            left_on=["eco_code"],
            right_on=["eco_code"],
        ),
        DeriveOpeningTransformation(),
        ApplyChessComOpeningLookupTransformation(),
    ]


class RawGamesToGamesStep(TransformStep):
    """Transform raw_games rows into enriched games rows for the current account."""

    def __init__(self, *, mode: PipelineMode = PipelineMode.INCREMENTAL) -> None:
        on, merge_strategy = preprocessing_transform_config(mode, incremental_on="game_id")
        super().__init__(
            name="RawGamesToGames",
            source_data_class=RawGame,
            target_data_class=Game,
            on=on,
            transformations=_raw_games_to_games_transformations(),
            loading_strategy=LoadingStrategy.MERGE,
            merge_strategy=merge_strategy,
        )


class ExtractUserMovesStep(TransformStep):
    """Extract user moves from games into games.moves for the current account."""

    def __init__(self, *, mode: PipelineMode = PipelineMode.INCREMENTAL) -> None:
        on, merge_strategy = preprocessing_transform_config(mode, incremental_on="game_id")
        super().__init__(
            name="ExtractUserMoves",
            source_data_class=Game,
            target_data_class=Move,
            on=on,
            source_columns=["game_id", "account_id", "cleaned_pgn", "color", "variant"],
            transformations=[
                ExtractUserMovesTransformation(),
                CreateHashedIdTransformation(data_class=Move),
            ],
            loading_strategy=LoadingStrategy.MERGE,
            merge_strategy=merge_strategy,
        )
