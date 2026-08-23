from chess_teacher.other.dataclasses import RawEcoCode
from chess_teacher.pipelines.ingestion.raw_games import RawGame
from chess_teacher.pipelines.modes import PipelineMode, preprocessing_transform_config
from chess_teacher.pipelines.preprocessing.games import Game
from chess_teacher.pipelines.preprocessing.move_characteristics import (
    AttackPressureTransformation,
    DiagonalOpennessTransformation,
    HangingValueTransformation,
    KingSafetyTransformation,
    LegalMovesTransformation,
    MaterialBalanceTransformation,
    MeanRankTransformation,
    MoveContextTransformation,
    MoveFlagsTransformation,
    PawnTensionTransformation,
    PinValueTransformation,
    StockfishEvaluationTransformation,
    VerticalOpennessTransformation,
)
from chess_teacher.pipelines.preprocessing.move_extraction import ExtractUserMovesTransformation
from chess_teacher.pipelines.preprocessing.moves import Move, MoveCharacteristics
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


class RawGamesToGamesStep(TransformStep):
    """Transform raw_games rows into enriched games rows for the current account."""

    def __init__(self, *, mode: PipelineMode = PipelineMode.INCREMENTAL) -> None:
        on, merge_strategy = preprocessing_transform_config(mode, incremental_on="game_id")
        super().__init__(
            name="RawGamesToGames",
            source_data_class=RawGame,
            target_data_class=Game,
            on=on,
            transformations=[
                ExpandRawResponseTransformation(),
                JoinWithTableTransformation(
                    with_data_class=Account,
                    columns=["account_id", "username", "platform"],
                ),
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
                    columns=["eco_code_id", "eco_code", "name", "pgn"],
                ),
                DeriveOpeningTransformation(),
                ApplyChessComOpeningLookupTransformation(),
            ],
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


class EnrichMoveCharacteristicsStep(TransformStep):
    """Compute move characteristics from moves into games.move_characteristics."""

    # FENs + UCI drive metrics; identity/opponent fields pass through to the target table.
    _SOURCE_COLUMNS: tuple[str, ...] = (
        "move_id",
        "game_id",
        "account_id",
        "fen_before",
        "fen_after",
        "move_uci",
        "previous_opponent_move_san",
        "previous_opponent_move_uci",
        "opponent_move_was_capture",
    )

    def __init__(self, *, mode: PipelineMode = PipelineMode.INCREMENTAL) -> None:
        on, merge_strategy = preprocessing_transform_config(mode, incremental_on="move_id")
        super().__init__(
            name="EnrichMoveCharacteristics",
            source_data_class=Move,
            target_data_class=MoveCharacteristics,
            on=on,
            source_columns=list(self._SOURCE_COLUMNS),
            transformations=[
                MaterialBalanceTransformation(),
                VerticalOpennessTransformation(),
                DiagonalOpennessTransformation(),
                PawnTensionTransformation(),
                LegalMovesTransformation(),
                KingSafetyTransformation(),
                MeanRankTransformation(),
                AttackPressureTransformation(),
                HangingValueTransformation(),
                PinValueTransformation(),
                MoveContextTransformation(),
                MoveFlagsTransformation(),
                StockfishEvaluationTransformation(depth=12, log_progress_percent=5),
            ],
            loading_strategy=LoadingStrategy.MERGE,
            merge_strategy=merge_strategy,
        )
