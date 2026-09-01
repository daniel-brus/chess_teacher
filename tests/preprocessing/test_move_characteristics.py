from __future__ import annotations

import chess
import polars as pl
import pytest

from chess_teacher.pipelines.preprocessing.fen_characteristic import (
    DualSidedFenCharacteristicTransformation,
    FenCharacteristicTransformation,
    _advance_fen_progress,
)
from chess_teacher.pipelines.preprocessing.move_characteristics.diagonal_openness import (
    DiagonalOpennessTransformation,
)
from chess_teacher.pipelines.preprocessing.move_characteristics.legal_moves import (
    LegalMovesTransformation,
)
from chess_teacher.pipelines.preprocessing.move_characteristics.material_balance import (
    MaterialBalanceTransformation,
)
from chess_teacher.pipelines.preprocessing.move_characteristics.move_context import (
    MoveContextTransformation,
)
from chess_teacher.pipelines.preprocessing.move_characteristics.move_flags import (
    MoveFlagsTransformation,
)
from chess_teacher.pipelines.preprocessing.move_characteristics.stockfish_evaluation import (
    StockfishEvaluationTransformation,
)
from chess_teacher.pipelines.preprocessing.move_characteristics.vertical_openness import (
    VerticalOpennessTransformation,
)
from chess_teacher.utils.chess_utils import (
    StockfishEngine,
    evaluation_to_white_pov_pawns,
    fen_diagonal_openness,
    fen_pawn_tension,
    fen_vertical_openness,
    game_over_white_pov_pawns,
    material_balance_white_pov,
    move_created_fork,
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


class _ConstantDualSidedCharacteristic(DualSidedFenCharacteristicTransformation):
    characteristic_name = "dual_sample"

    def __init__(self, value_by_fen: dict[str, tuple[float, float]]) -> None:
        super().__init__(log_progress_percent=None, n_workers=1)
        self.value_by_fen = value_by_fen

    def evaluate_sides(self, fen: str, *, row: dict[str, object]) -> tuple[float, float]:
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
        "previous_opponent_move_san": [None],
        "previous_opponent_move_uci": [None],
        "opponent_move_was_capture": [False],
    })


def test_fen_characteristic_checkpoint_builds_partial_rows() -> None:
    transformation = _ConstantCharacteristic({_START: 2.0, _AFTER_E4: 0.5})
    transformation.checkpoint_percent = 10
    transformation._prepare_checkpoint_rows(_sample_moves_df())
    merged: list[pl.DataFrame] = []

    class _FakeClient:
        def merge(self, df: pl.DataFrame, table_metadata: object, *, strategy: object) -> None:
            del table_metadata, strategy
            merged.append(df)

    transformation._db_client = _FakeClient()  # type: ignore[assignment]
    transformation._table_metadata = object()  # type: ignore[assignment]
    transformation._maybe_checkpoint_fen_batch({_START: 2.0})

    assert len(merged) == 1
    row = merged[0].to_dicts()[0]
    assert row["move_id"] == "m1"
    assert row["sample_before"] == 2.0
    assert "sample_after" not in row
    assert "sample_delta" not in row

    transformation._maybe_checkpoint_fen_batch({_AFTER_E4: 0.5})
    assert len(merged) == 2
    row = merged[1].to_dicts()[0]
    assert row["sample_before"] == 2.0
    assert row["sample_after"] == 0.5
    assert row["sample_delta"] == pytest.approx(-1.5)


def test_fen_characteristic_after_and_delta() -> None:
    transformation = _ConstantCharacteristic({_START: 2.0, _AFTER_E4: 0.5})
    result = transformation.transform(_sample_moves_df())
    assert result["sample_before"][0] == 2.0
    assert result["sample_after"][0] == 0.5
    assert result["sample_delta"][0] == pytest.approx(-1.5)


def test_fen_characteristic_raises_when_evaluate_fails() -> None:
    transformation = _ConstantCharacteristic({_START: 2.0})
    with pytest.raises(TransformationError, match="No value for FEN"):
        transformation.transform(_sample_moves_df())


def test_dual_sided_fen_characteristic_after_and_delta() -> None:
    transformation = _ConstantDualSidedCharacteristic({
        _START: (20.0, 20.0),
        _AFTER_E4: (22.0, 18.0),
    })
    result = transformation.transform(_sample_moves_df())
    assert result["white_dual_sample_after"][0] == 22.0
    assert result["white_dual_sample_delta"][0] == pytest.approx(2.0)
    assert result["black_dual_sample_after"][0] == 18.0
    assert result["black_dual_sample_delta"][0] == pytest.approx(-2.0)


def test_vertical_openness_starting_position_is_zero() -> None:
    assert fen_vertical_openness(_START) == 0.0


def test_diagonal_openness_starting_position_is_zero() -> None:
    assert fen_diagonal_openness(_START) == 0.0


def test_vertical_openness_transformation() -> None:
    result = VerticalOpennessTransformation().transform(_sample_moves_df())
    assert result["vertical_openness_after"][0] == 0.0
    assert result["vertical_openness_delta"][0] == 0.0


def test_diagonal_openness_transformation() -> None:
    result = DiagonalOpennessTransformation().transform(_sample_moves_df())
    assert result["diagonal_openness_after"][0] == 0.0
    assert result["diagonal_openness_delta"][0] == 0.0


def test_pawn_tension_starting_position() -> None:
    assert fen_pawn_tension(_START) == 0.0


def test_legal_moves_transformation() -> None:
    result = LegalMovesTransformation().transform(_sample_moves_df())
    assert result["white_legal_moves_after"][0] == pytest.approx(30.0)
    assert result["black_legal_moves_after"][0] == pytest.approx(20.0)


def test_move_flags_opening_e4() -> None:
    result = MoveFlagsTransformation().transform(_sample_moves_df())
    assert result["is_capture"][0] is False
    assert result["is_castle"][0] is False
    assert result["gave_check"][0] is False
    assert result["created_fork"][0] is False


def test_move_flags_capture() -> None:
    # 1. e4 d5 — white captures with exd5
    fen_before = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    fen_after = "rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2"
    df = pl.DataFrame({
        "fen_before": [fen_before],
        "fen_after": [fen_after],
        "move_uci": ["e4d5"],
        "opponent_move_was_capture": [False],
    })
    result = MoveFlagsTransformation().transform(df)
    assert result["is_capture"][0] is True


def test_move_created_fork_loose_pawns() -> None:
    fen_before = "8/p1p5/8/8/3N4/8/8/4K2k w - - 0 1"
    board = chess.Board(fen_before)
    board.push(chess.Move.from_uci("d4b5"))
    assert move_created_fork(fen_before, board.fen(), "d4b5") is True


def test_move_created_fork_defended_pawns_not_counted() -> None:
    fen_before = "8/1pp5/p1p5/8/3N4/8/8/4K2k w - - 0 1"
    board = chess.Board(fen_before)
    board.push(chess.Move.from_uci("d4b5"))
    assert move_created_fork(fen_before, board.fen(), "d4b5") is False


def test_move_created_fork_royal_fork() -> None:
    fen_before = "3k3q/8/8/6N1/8/8/8/4K3 w - - 0 1"
    board = chess.Board(fen_before)
    board.push(chess.Move.from_uci("g5f7"))
    assert move_created_fork(fen_before, board.fen(), "g5f7") is True


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
    assert result["material_balance_before"][0] == 0.0
    assert result["material_balance_after"][0] == 0.0
    assert result["material_balance_delta"][0] == 0.0


def test_move_context_opening_first_move() -> None:
    result = MoveContextTransformation().transform(_sample_moves_df())
    assert result["is_in_check_before"][0] is False
    assert result["is_opening"][0] is True
    assert result["is_middle_game"][0] is False
    assert result["is_end_game"][0] is False
    assert result["has_castling_rights_before"][0] is True


def test_move_flags_recapture() -> None:
    fen_before = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    fen_after = "rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2"
    df = pl.DataFrame({
        "fen_before": [fen_before],
        "fen_after": [fen_after],
        "move_uci": ["e4d5"],
        "opponent_move_was_capture": [False],
    })
    result = MoveFlagsTransformation().transform(df)
    assert result["is_capture"][0] is True
    assert result["is_recapture"][0] is False

    df_recapture = df.with_columns(pl.lit(True).alias("opponent_move_was_capture"))
    result_recapture = MoveFlagsTransformation().transform(df_recapture)
    assert result_recapture["is_recapture"][0] is True


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
    assert result["evaluation_before"][0] == pytest.approx(0.2)
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
