import pytest

from chess_teacher.pipelines.ingestion.raw_games import RawGame
from chess_teacher.pipelines.modes import PipelineMode, preprocessing_transform_config
from chess_teacher.pipelines.preprocessing.games import Game
from chess_teacher.utils.db.client import MergeStrategy
from chess_teacher.utils.pipeline_utils.pipeline_steps import LoadingStrategy, TransformStep


def test_preprocessing_transform_config_incremental() -> None:
    on, merge = preprocessing_transform_config(
        PipelineMode.INCREMENTAL,
        incremental_on="game_id",
    )
    assert on == "game_id"
    assert merge == MergeStrategy.upsert()


def test_preprocessing_transform_config_retry() -> None:
    on, merge = preprocessing_transform_config(
        PipelineMode.RETRY,
        incremental_on="game_id",
    )
    assert on == "game_id"
    assert merge == MergeStrategy.upsert()


def test_preprocessing_transform_config_reprocess() -> None:
    on, merge = preprocessing_transform_config(
        PipelineMode.REPROCESS,
        incremental_on="game_id",
    )
    assert on is None
    assert merge == MergeStrategy.upsert()


def test_preprocessing_transform_config_full_reload() -> None:
    on, merge = preprocessing_transform_config(
        PipelineMode.FULL_RELOAD,
        incremental_on="game_id",
    )
    assert on is None
    assert merge == MergeStrategy.full_sync()


def test_pipeline_mode_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="'sync'"):
        PipelineMode("sync")


def test_transform_step_rejects_incremental_on_with_full_sync() -> None:
    with pytest.raises(ValueError, match="full_sync"):
        TransformStep(
            name="TestStep",
            source_data_class=RawGame,
            target_data_class=Game,
            on="game_id",
            loading_strategy=LoadingStrategy.MERGE,
            merge_strategy=MergeStrategy.full_sync(),
        )
