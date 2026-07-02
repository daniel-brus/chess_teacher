from __future__ import annotations

import chess
import polars as pl
import pytest

from chess_teacher.pipelines.preprocessing.fen_characteristic import (
    FenCharacteristicTransformation,
    _advance_fen_progress,
)
from chess_teacher.pipelines.preprocessing.move_characteristics.material_balance import (
    MaterialBalanceTransformation,
)
from chess_teacher.pipelines.preprocessing.move_characteristics.stockfish_evaluation import (
    StockfishEvaluationTransformation,
)
from chess_teacher.utils.chess_utils import (
    StockfishEngine,
    evaluation_to_white_pov_pawns,
    game_over_white_pov_pawns,
    material_balance_white_pov,
)
from chess_teacher.utils.exception_utils import TransformationError

_START = chess.STARTING_FEN
_AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
_WHITE_UP_PAWN = "rnbqkbnr/ppp1pppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
_CHECKMATE_WHITE_WINS = "5k1Q/1p4R1/3p4/8/1P6/7P/5P1K/8 b - - 2 45"


class _ConstantCharacteristic(FenCharacteristicTransformation):
    characteristic_name = "sample"

    def __init__(self, value_by_fen: dict[str, float]) -> None:
        super().__init__(log_progress_percent=None, n_workers=1)
        self.value_by_fen = value_by_fen

    def evaluate(self, fen: str, *, row: dict[str, object]) -> float:
        del row
        if fen not in self.value_by_fen:
            raise TransformationError(f"No value for FEN: {fen!r}")
        return self.value_by_fen[fen]


def _sample_moves_df() -> pl.DataFrame:
    return pl.DataFrame({
        "move_id": ["m1"],
        "game_id": ["g1"],
        "account_id": ["a1"],
        "move_nr": [1],
        "ply": [1],
        "move_san": ["e4"],
        "move_uci": ["e2e4"],
        "fen_before": [_START],
        "fen_after": [_AFTER_E4],
    })


def test_fen_characteristic_after_and_delta() -> None:
    transformation = _ConstantCharacteristic({_START: 2.0, _AFTER_E4: 0.5})
    result = transformation.transform(_sample_moves_df())
    assert result["sample_after"][0] == 0.5
    assert result["sample_delta"][0] == pytest.approx(-1.5)


def test_fen_characteristic_raises_when_evaluate_fails() -> None:
    transformation = _ConstantCharacteristic({_START: 2.0})
    with pytest.raises(TransformationError, match="No value for FEN"):
        transformation.transform(_sample_moves_df())


def test_advance_fen_progress_emits_every_five_percent() -> None:
    messages: list[tuple[int, int]] = []

    def report(completed: int, total: int) -> None:
        messages.append((completed, total))

    last = _advance_fen_progress(
        completed=1,
        total=20,
        progress_percent=5,
        last_logged_percent=0,
        report=report,
    )
    assert messages == [(1, 20)]
    assert last == 5

    last = _advance_fen_progress(
        completed=20,
        total=20,
        progress_percent=5,
        last_logged_percent=last,
        report=report,
    )
    assert messages[-1] == (20, 20)
    assert last == 100


def test_material_balance_starting_position_is_zero() -> None:
    assert material_balance_white_pov(_START) == 0.0


def test_material_balance_white_up_pawn() -> None:
    assert material_balance_white_pov(_WHITE_UP_PAWN) == 1.0


def test_material_balance_transformation() -> None:
    result = MaterialBalanceTransformation().transform(_sample_moves_df())
    assert result["material_balance_after"][0] == 0.0
    assert result["material_balance_delta"][0] == 0.0


def test_evaluation_to_white_pov_cp_white_to_move() -> None:
    score = evaluation_to_white_pov_pawns({"type": "cp", "value": 50}, _START)
    assert score == pytest.approx(0.5)


def test_evaluation_to_white_pov_cp_black_to_move() -> None:
    score = evaluation_to_white_pov_pawns({"type": "cp", "value": 50}, _AFTER_E4)
    assert score == pytest.approx(-0.5)


def test_evaluation_to_white_pov_mate_white_winning() -> None:
    score = evaluation_to_white_pov_pawns({"type": "mate", "value": 3}, _START)
    assert score == pytest.approx(97.0)


def test_game_over_white_pov_checkmate_white_wins() -> None:
    board = chess.Board(_CHECKMATE_WHITE_WINS)
    assert board.is_checkmate()
    assert game_over_white_pov_pawns(board) == pytest.approx(100.0)


def test_stockfish_evaluation_transformation_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluate_calls: list[str] = []

    class _FakeEngine:
        def __init__(self, *, depth: int = 20, path: str | None = None) -> None:
            del depth, path

        def __enter__(self) -> _FakeEngine:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def evaluate_white_pov_pawns(self, fen: str) -> float | None:
            evaluate_calls.append(fen)
            if fen == _START:
                return 0.2
            if fen == _AFTER_E4:
                return 0.05
            return None

    monkeypatch.setattr(
        "chess_teacher.pipelines.preprocessing.move_characteristics.stockfish_evaluation.StockfishEngine",
        _FakeEngine,
    )
    result = StockfishEvaluationTransformation(
        depth=20, stockfish_path="fake", n_workers=1
    ).transform(_sample_moves_df())
    assert result["evaluation_after"][0] == pytest.approx(0.05)
    assert result["evaluation_delta"][0] == pytest.approx(-0.15)
    assert set(evaluate_calls) == {_START, _AFTER_E4}


def test_stockfish_evaluation_dedupes_shared_fens(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluate_calls: list[str] = []

    class _FakeEngine:
        def __init__(self, *, depth: int = 20, path: str | None = None) -> None:
            del depth, path

        def __enter__(self) -> _FakeEngine:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def evaluate_white_pov_pawns(self, fen: str) -> float | None:
            evaluate_calls.append(fen)
            return 0.0

    monkeypatch.setattr(
        "chess_teacher.pipelines.preprocessing.move_characteristics.stockfish_evaluation.StockfishEngine",
        _FakeEngine,
    )
    df = pl.DataFrame({
        "fen_before": [_START, _AFTER_E4],
        "fen_after": [_AFTER_E4, _WHITE_UP_PAWN],
    })
    StockfishEvaluationTransformation(depth=20, stockfish_path="fake", n_workers=1).transform(df)
    assert set(evaluate_calls) == {_START, _AFTER_E4, _WHITE_UP_PAWN}


def test_stockfish_engine_game_over_without_active_engine() -> None:
    engine = StockfishEngine(path="nonexistent-path")
    score = engine.evaluate_white_pov_pawns(_CHECKMATE_WHITE_WINS)
    assert score == pytest.approx(100.0)


def test_stockfish_evaluation_handles_checkmate_fen(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeEngine:
        def __init__(self, *, depth: int = 20, path: str | None = None) -> None:
            del depth, path

        def __enter__(self) -> _FakeEngine:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def evaluate_white_pov_pawns(self, fen: str) -> float | None:
            if fen == _CHECKMATE_WHITE_WINS:
                return game_over_white_pov_pawns(chess.Board(fen))
            return 0.0

    monkeypatch.setattr(
        "chess_teacher.pipelines.preprocessing.move_characteristics.stockfish_evaluation.StockfishEngine",
        _FakeEngine,
    )
    df = pl.DataFrame({
        "fen_before": [_START],
        "fen_after": [_CHECKMATE_WHITE_WINS],
    })
    result = StockfishEvaluationTransformation(
        depth=20, stockfish_path="fake", n_workers=1
    ).transform(df)
    assert result["evaluation_after"][0] == pytest.approx(100.0)
    assert result["evaluation_delta"][0] == pytest.approx(100.0)
