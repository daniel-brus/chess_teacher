"""Heavy integration: preprocessing TransformSteps x PipelineMode (mocked DB / Stockfish)."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from unittest.mock import MagicMock

import chess
import polars as pl
import pytest

from chess_teacher.pipelines.modes import (
    PIPELINE_MODES,
    PipelineMode,
    preprocessing_transform_config,
)
from chess_teacher.pipelines.preprocessing.move_characteristics.candidate_evaluations import (
    CandidateEvaluationsTransformation,
)
from chess_teacher.pipelines.preprocessing.move_characteristics.stockfish_evaluation import (
    StockfishEvaluationTransformation,
)
from chess_teacher.pipelines.preprocessing.opening_tables import RawEcoCode
from chess_teacher.pipelines.preprocessing.pipeline_steps import (
    EnrichCheapMoveCharacteristicsStep,
    EnrichExpensiveMoveCharacteristicsStep,
    ExtractUserMovesStep,
    RawGamesToGamesStep,
)
from chess_teacher.platform.account import Account, AccountPlatform
from chess_teacher.utils.chess_utils import Color
from chess_teacher.utils.db.client import MergeStrategy, WriteResult, WriteStrategy
from chess_teacher.utils.metadata_utils import TableMetadata
from chess_teacher.utils.pipeline_utils.pipeline_base import PipelineContext
from chess_teacher.utils.pipeline_utils.pipeline_steps import TransformStep
from chess_teacher.utils.pipeline_utils.transformations import JoinWithTableTransformation

_ACCOUNT_ID = "acct-1"
_USER = "TestPlayer"
_START = chess.STARTING_FEN
_AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
_SAMPLE_PGN = "1. e4 e5 2. Nf3 Nc6 3. d3"
_INGESTED_AT = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)


@dataclass(frozen=True)
class CapturedSave:
    df: pl.DataFrame | None
    merge_strategy: MergeStrategy
    match_condition: str | None
    on: str | None


def _mock_db(*, existing_keys: list[str] | None = None) -> MagicMock:
    db = MagicMock()
    db.ensure_metadata.return_value = None
    db.table_exists.return_value = True
    keys = existing_keys or []

    def _existing_query(sql: str, _params: dict) -> list[dict]:
        if not keys:
            return []
        if '"move_id"' in sql or "move_id" in sql:
            return [{"move_id": k} for k in keys]
        if '"game_id"' in sql or "game_id" in sql:
            return [{"game_id": k} for k in keys]
        return []

    db.engine.execute_parameterized_query.side_effect = _existing_query
    return db


def _capture_run(
    monkeypatch: pytest.MonkeyPatch,
    step: TransformStep,
    source_df: pl.DataFrame,
    *,
    existing_keys: list[str] | None = None,
    prepare: Callable[[TransformStep, MagicMock], None] | None = None,
) -> CapturedSave:
    db = _mock_db(existing_keys=existing_keys)
    if prepare is not None:
        prepare(step, db)

    saved: list[pl.DataFrame] = []

    def capture_save(
        _db: MagicMock,
        _meta: object,
        data: pl.DataFrame,
    ) -> WriteResult:
        saved.append(data)
        return WriteResult(strategy=WriteStrategy.MERGE, rows_inserted=data.height)

    monkeypatch.setattr(step, "_load_records", lambda _db, _ctx: source_df.clone())
    monkeypatch.setattr(step, "_save_records", capture_save)

    step.run(db, PipelineContext(user_id="u1", account_id=_ACCOUNT_ID))

    return CapturedSave(
        df=saved[0] if saved else None,
        merge_strategy=step.merge_strategy,
        match_condition=step.match_condition,
        on=step.on,
    )


def _stub_stockfish(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_transform(self: StockfishEvaluationTransformation, df: pl.DataFrame) -> pl.DataFrame:
        del self
        return df.with_columns(
            evaluation_before=pl.lit(0.10),
            evaluation_after=pl.lit(0.25),
            evaluation_delta=pl.lit(0.15),
        )

    def fake_candidates(self: CandidateEvaluationsTransformation, df: pl.DataFrame) -> pl.DataFrame:
        del self
        payload = {"version": 1, "depth": 12, "evals": {"e2e4": 0.2}}
        return df.with_columns(
            pl.Series("candidate_evaluations", [payload] * df.height, dtype=pl.Object),
        )

    monkeypatch.setattr(StockfishEvaluationTransformation, "transform", fake_transform)
    monkeypatch.setattr(CandidateEvaluationsTransformation, "transform", fake_candidates)


# --- fixtures -----------------------------------------------------------------

# Three games: white, black, non-standard (skipped by ExtractUserMoves).
_DEFAULT_GAME_IDS = ["game-1", "game-2", "game-3"]
_DEFAULT_MOVE_IDS = ["move-1", "move-2", "move-3"]


def _games_source(*, game_ids: list[str] | None = None) -> pl.DataFrame:
    ids = _DEFAULT_GAME_IDS if game_ids is None else game_ids
    n = len(ids)
    colors = [Color.WHITE.value, Color.BLACK.value, Color.WHITE.value][:n]
    while len(colors) < n:
        colors.append(Color.WHITE.value)
    variants = ["standard", "standard", "crazyhouse"][:n]
    while len(variants) < n:
        variants.append("standard")
    pgns = [_SAMPLE_PGN, _SAMPLE_PGN, _SAMPLE_PGN][:n]
    while len(pgns) < n:
        pgns.append(_SAMPLE_PGN)
    return pl.DataFrame({
        "game_id": ids,
        "account_id": [_ACCOUNT_ID] * n,
        "cleaned_pgn": pgns,
        "color": colors,
        "variant": variants,
    })


def _moves_source(*, move_ids: list[str] | None = None) -> pl.DataFrame:
    ids = _DEFAULT_MOVE_IDS if move_ids is None else move_ids
    n = len(ids)
    return pl.DataFrame({
        "move_id": ids,
        "game_id": [f"game-{(i % 3) + 1}" for i in range(n)],
        "account_id": [_ACCOUNT_ID] * n,
        "fen_before": [_START] * n,
        "fen_after": [_AFTER_E4] * n,
        "move_uci": ["e2e4"] * n,
        "previous_opponent_move_san": [None] * n,
        "previous_opponent_move_uci": [None] * n,
        "opponent_move_was_capture": [False] * n,
    })


def _lichess_raw_response(
    *,
    platform_game_id: str = "lich-1",
    color: str = Color.WHITE.value,
    pgn: str | None = None,
    include_pgn: bool = True,
) -> str:
    movetext = (
        '[Event "Casual"]\n'
        '[Result "1-0"]\n'
        '[Termination "Normal"]\n'
        "\n"
        "1. e4 e5 2. Nf3 Nc6 3. d3 1-0\n"
    )
    if pgn is not None:
        movetext = pgn
    if color == Color.WHITE.value:
        players = {
            "white": {"user": {"name": _USER}, "rating": 1500},
            "black": {"user": {"name": "Opponent"}, "rating": 1480},
        }
        winner = "white"
    else:
        players = {
            "white": {"user": {"name": "Opponent"}, "rating": 1480},
            "black": {"user": {"name": _USER}, "rating": 1500},
        }
        winner = "black"
    payload: dict = {
        "id": platform_game_id,
        "variant": "standard",
        "status": "resign",
        "winner": winner,
        "createdAt": 1_700_000_000_000,
        "lastMoveAt": 1_700_000_100_000,
        "clock": {"initial": 300, "increment": 0},
        "opening": {"eco": "C44", "name": "King's Pawn Game: Ponziani Opening"},
        "players": players,
    }
    if include_pgn:
        payload["pgn"] = movetext
    else:
        # Column must exist so FilterGamesWithPGN can drop empty rows.
        payload["pgn"] = ""
    return json.dumps(payload)


def _raw_games_source(*, game_ids: list[str] | None = None) -> pl.DataFrame:
    """Three raw games: white, black, missing PGN (dropped by FilterGamesWithPGN)."""
    ids = _DEFAULT_GAME_IDS if game_ids is None else game_ids
    responses = [
        _lichess_raw_response(platform_game_id="lich-0", color=Color.WHITE.value),
        _lichess_raw_response(platform_game_id="lich-1", color=Color.BLACK.value),
        _lichess_raw_response(platform_game_id="lich-2", include_pgn=False),
    ]
    while len(responses) < len(ids):
        responses.append(_lichess_raw_response(platform_game_id=f"lich-{len(responses)}"))
    responses = responses[: len(ids)]
    return pl.DataFrame({
        "game_id": ids,
        "platform_game_id": [f"lich-{i}" for i in range(len(ids))],
        "account_id": [_ACCOUNT_ID] * len(ids),
        "raw_response": responses,
        "source_file": [f"ingested/{_ACCOUNT_ID}/a.jsonl"] * len(ids),
        "ingested_at": [_INGESTED_AT] * len(ids),
    })


def _wire_raw_games_joins(step: TransformStep, db: MagicMock) -> None:
    account_meta = Account.get_metadata()
    eco_meta = RawEcoCode.get_metadata()

    def fake_read(
        table: TableMetadata,
        columns: list[str] | None = None,
        where: str | None = None,
        as_polars: bool = True,
    ) -> pl.DataFrame:
        del where, as_polars
        if table.qualified_name_sql() == account_meta.qualified_name_sql():
            frame = pl.DataFrame({
                "account_id": [_ACCOUNT_ID],
                "username": [_USER],
                "platform": [AccountPlatform.LICHESS.value],
            })
        elif table.qualified_name_sql() == eco_meta.qualified_name_sql():
            frame = pl.DataFrame({
                "eco_code_id": ["eco-c44"],
                "eco_code": ["C44"],
                "name": ["King's Pawn Game: Ponziani Opening"],
                "pgn": ["1. e4 e5 2. Nf3 Nc6 3. c3"],
            })
        else:
            raise AssertionError(f"Unexpected join table: {table.qualified_name_sql()}")
        if columns is not None:
            keep = [c for c in columns if c in frame.columns]
            return frame.select(keep)
        return frame

    db.read.side_effect = fake_read
    for transformation in step.transformations:
        if isinstance(transformation, JoinWithTableTransformation):
            transformation.db_client = db


# --- config contracts ---------------------------------------------------------


@pytest.mark.parametrize("mode", PIPELINE_MODES)
def test_extract_user_moves_mode_config(mode: PipelineMode) -> None:
    expected_on, expected_merge = preprocessing_transform_config(mode, incremental_on="game_id")
    step = ExtractUserMovesStep(mode=mode)
    assert step.on == expected_on
    assert step.merge_strategy == expected_merge


@pytest.mark.parametrize("mode", PIPELINE_MODES)
def test_cheap_enrich_mode_config(mode: PipelineMode) -> None:
    expected_on, expected_merge = preprocessing_transform_config(mode, incremental_on="move_id")
    step = EnrichCheapMoveCharacteristicsStep(mode=mode)
    assert step.on == expected_on
    assert step.merge_strategy == expected_merge


@pytest.mark.parametrize("mode", PIPELINE_MODES)
def test_expensive_enrich_ignores_mode_for_write_path(mode: PipelineMode) -> None:
    step = EnrichExpensiveMoveCharacteristicsStep(mode=mode)
    assert step.on is None
    assert step.merge_strategy == MergeStrategy.upsert()
    assert step.merge_strategy.when_not_matched_by_source == "ignore"


@pytest.mark.parametrize("mode", PIPELINE_MODES)
def test_raw_games_to_games_mode_config(mode: PipelineMode) -> None:
    expected_on, expected_merge = preprocessing_transform_config(mode, incremental_on="game_id")
    step = RawGamesToGamesStep(mode=mode)
    assert step.on == expected_on
    assert step.merge_strategy == expected_merge


# --- ExtractUserMoves ---------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "existing_keys", "expected_game_ids", "expected_rows"),
    [
        # game-3 is crazyhouse → extract yields no rows for it
        (PipelineMode.INCREMENTAL, [], ["game-1", "game-2"], 5),  # white 3 + black 2
        (PipelineMode.RETRY, ["game-1"], ["game-2"], 2),
        (PipelineMode.REPROCESS, ["game-1", "game-2", "game-3"], ["game-1", "game-2"], 5),
        (PipelineMode.FULL_RELOAD, ["game-1"], ["game-1", "game-2"], 5),
    ],
)
def test_extract_user_moves_modes(
    monkeypatch: pytest.MonkeyPatch,
    mode: PipelineMode,
    existing_keys: list[str],
    expected_game_ids: list[str],
    expected_rows: int,
) -> None:
    step = ExtractUserMovesStep(mode=mode)
    captured = _capture_run(
        monkeypatch,
        step,
        _games_source(),
        existing_keys=existing_keys,
    )
    assert captured.df is not None
    assert sorted(captured.df["game_id"].unique().to_list()) == expected_game_ids
    assert captured.df.height == expected_rows
    assert "move_id" in captured.df.columns
    assert "fen_before" in captured.df.columns
    if mode == PipelineMode.FULL_RELOAD:
        assert captured.merge_strategy.when_not_matched_by_source == "delete"
        assert captured.match_condition is not None
        assert _ACCOUNT_ID in captured.match_condition


def test_extract_user_moves_incremental_all_existing_skips_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = ExtractUserMovesStep(mode=PipelineMode.INCREMENTAL)
    captured = _capture_run(
        monkeypatch,
        step,
        _games_source(),
        existing_keys=list(_DEFAULT_GAME_IDS),
    )
    assert captured.df is None


def test_extract_user_moves_empty_source_skips_save(monkeypatch: pytest.MonkeyPatch) -> None:
    step = ExtractUserMovesStep(mode=PipelineMode.INCREMENTAL)
    captured = _capture_run(monkeypatch, step, _games_source(game_ids=[]))
    assert captured.df is None


# --- EnrichCheap --------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "existing_keys", "expected_move_ids"),
    [
        (PipelineMode.INCREMENTAL, [], ["move-1", "move-2", "move-3"]),
        (PipelineMode.RETRY, ["move-1"], ["move-2", "move-3"]),
        (PipelineMode.REPROCESS, ["move-1", "move-2", "move-3"], ["move-1", "move-2", "move-3"]),
        (PipelineMode.FULL_RELOAD, ["move-1"], ["move-1", "move-2", "move-3"]),
    ],
)
def test_cheap_enrich_modes(
    monkeypatch: pytest.MonkeyPatch,
    mode: PipelineMode,
    existing_keys: list[str],
    expected_move_ids: list[str],
) -> None:
    step = EnrichCheapMoveCharacteristicsStep(mode=mode)
    captured = _capture_run(
        monkeypatch,
        step,
        _moves_source(),
        existing_keys=existing_keys,
    )
    assert captured.df is not None
    assert sorted(captured.df["move_id"].to_list()) == expected_move_ids
    assert "material_balance_after" in captured.df.columns
    assert "is_capture" in captured.df.columns
    assert "evaluation_after" not in captured.df.columns
    assert "candidate_evaluations" not in captured.df.columns
    if mode == PipelineMode.FULL_RELOAD:
        assert captured.merge_strategy.when_not_matched_by_source == "delete"
        assert captured.match_condition is not None
        assert _ACCOUNT_ID in captured.match_condition


def test_cheap_enrich_incremental_all_existing_skips_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = EnrichCheapMoveCharacteristicsStep(mode=PipelineMode.INCREMENTAL)
    captured = _capture_run(
        monkeypatch,
        step,
        _moves_source(),
        existing_keys=list(_DEFAULT_MOVE_IDS),
    )
    assert captured.df is None


# --- EnrichExpensive ----------------------------------------------------------


@pytest.mark.parametrize("mode", PIPELINE_MODES)
def test_expensive_enrich_modes_run_transforms(
    monkeypatch: pytest.MonkeyPatch,
    mode: PipelineMode,
) -> None:
    _stub_stockfish(monkeypatch)
    step = EnrichExpensiveMoveCharacteristicsStep(mode=mode)
    # Expensive step never incremental-filters on move_id; all loaded rows process.
    captured = _capture_run(
        monkeypatch,
        step,
        _moves_source(),
        existing_keys=list(_DEFAULT_MOVE_IDS),
    )
    assert captured.df is not None
    assert captured.df.height == 3
    assert captured.on is None
    assert captured.merge_strategy == MergeStrategy.upsert()
    assert captured.df["evaluation_after"].to_list() == [0.25, 0.25, 0.25]
    assert all(v is not None for v in captured.df["candidate_evaluations"].to_list())
    assert captured.merge_strategy.when_not_matched_by_source == "ignore"


def test_expensive_enrich_empty_source_skips_save(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_stockfish(monkeypatch)
    step = EnrichExpensiveMoveCharacteristicsStep(mode=PipelineMode.INCREMENTAL)
    captured = _capture_run(monkeypatch, step, _moves_source(move_ids=[]))
    assert captured.df is None


# --- RawGamesToGames ----------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "existing_keys", "expected_game_ids", "expected_colors"),
    [
        # game-3 has no PGN → dropped after filter
        (
            PipelineMode.INCREMENTAL,
            [],
            ["game-1", "game-2"],
            [Color.WHITE.value, Color.BLACK.value],
        ),
        (PipelineMode.RETRY, ["game-1"], ["game-2"], [Color.BLACK.value]),
        (
            PipelineMode.REPROCESS,
            ["game-1"],
            ["game-1", "game-2"],
            [Color.WHITE.value, Color.BLACK.value],
        ),
        (
            PipelineMode.FULL_RELOAD,
            ["game-1", "game-2", "game-3"],
            ["game-1", "game-2"],
            [Color.WHITE.value, Color.BLACK.value],
        ),
    ],
)
def test_raw_games_to_games_modes(
    monkeypatch: pytest.MonkeyPatch,
    mode: PipelineMode,
    existing_keys: list[str],
    expected_game_ids: list[str],
    expected_colors: list[str],
) -> None:
    monkeypatch.setattr(
        "chess_teacher.utils.pipeline_utils.transformations.get_db_client",
        lambda: MagicMock(),
    )
    monkeypatch.setattr(
        "chess_teacher.pipelines.preprocessing.chess_com_openings.load_slug_title_lookup",
        lambda: {},
    )

    step = RawGamesToGamesStep(mode=mode)
    captured = _capture_run(
        monkeypatch,
        step,
        _raw_games_source(),
        existing_keys=existing_keys,
        prepare=_wire_raw_games_joins,
    )
    assert captured.df is not None
    assert sorted(captured.df["game_id"].to_list()) == expected_game_ids
    color_by_game = {
        row["game_id"]: row["color"]
        for row in captured.df.select(["game_id", "color"]).iter_rows(named=True)
    }
    assert [color_by_game[gid] for gid in expected_game_ids] == expected_colors
    assert captured.df["cleaned_pgn"].null_count() == 0
    assert all(
        name == "King's Pawn Game: Ponziani Opening"
        for name in captured.df["opening_name"].to_list()
    )
    if mode == PipelineMode.FULL_RELOAD:
        assert captured.merge_strategy.when_not_matched_by_source == "delete"
        assert captured.match_condition is not None


def test_raw_games_incremental_all_existing_skips_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "chess_teacher.utils.pipeline_utils.transformations.get_db_client",
        lambda: MagicMock(),
    )
    step = RawGamesToGamesStep(mode=PipelineMode.INCREMENTAL)
    captured = _capture_run(
        monkeypatch,
        step,
        _raw_games_source(),
        existing_keys=list(_DEFAULT_GAME_IDS),
        prepare=_wire_raw_games_joins,
    )
    assert captured.df is None


def test_raw_games_only_no_pgn_rows_skips_save(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "chess_teacher.utils.pipeline_utils.transformations.get_db_client",
        lambda: MagicMock(),
    )
    monkeypatch.setattr(
        "chess_teacher.pipelines.preprocessing.chess_com_openings.load_slug_title_lookup",
        lambda: {},
    )
    step = RawGamesToGamesStep(mode=PipelineMode.INCREMENTAL)
    source = _raw_games_source(game_ids=["game-only"])
    # Force the single row to be the no-PGN payload
    source = source.with_columns(raw_response=pl.Series([_lichess_raw_response(include_pgn=False)]))
    captured = _capture_run(
        monkeypatch,
        step,
        source,
        prepare=_wire_raw_games_joins,
    )
    assert captured.df is None
