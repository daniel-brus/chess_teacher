"""Behavioral tests for move_characteristics metrics using curated positions."""

from __future__ import annotations

import chess
import polars as pl
import pytest

from chess_teacher.pipelines.preprocessing.move_characteristics.attack_pressure import (
    AttackPressureTransformation,
)
from chess_teacher.pipelines.preprocessing.move_characteristics.diagonal_openness import (
    DiagonalOpennessTransformation,
)
from chess_teacher.pipelines.preprocessing.move_characteristics.hanging_value import (
    HangingValueTransformation,
)
from chess_teacher.pipelines.preprocessing.move_characteristics.king_safety import (
    KingSafetyTransformation,
)
from chess_teacher.pipelines.preprocessing.move_characteristics.mean_rank import (
    MeanRankTransformation,
)
from chess_teacher.pipelines.preprocessing.move_characteristics.move_context import (
    MoveContextTransformation,
)
from chess_teacher.pipelines.preprocessing.move_characteristics.move_flags import (
    MoveFlagsTransformation,
)
from chess_teacher.pipelines.preprocessing.move_characteristics.pawn_tension import (
    PawnTensionTransformation,
)
from chess_teacher.pipelines.preprocessing.move_characteristics.pin_value import (
    PinValueTransformation,
)
from chess_teacher.pipelines.preprocessing.move_characteristics.vertical_openness import (
    VerticalOpennessTransformation,
)
from chess_teacher.utils.chess_utils import (
    fen_attack_pressure,
    fen_diagonal_openness,
    fen_game_phase,
    fen_hanging_value,
    fen_has_castling_rights,
    fen_has_hanging_piece,
    fen_is_in_check,
    fen_king_safety,
    fen_mean_rank,
    fen_pawn_tension,
    fen_pin_value,
    fen_vertical_openness,
    move_gave_check,
    move_is_castle,
    move_is_en_passant,
    move_is_promotion,
)

_START = chess.STARTING_FEN
_KINGS_ONLY = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
_E4_D5 = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
_HANGING_WHITE_KNIGHT = "4k3/8/4p3/3N4/8/8/8/4K3 w - - 0 1"
_PINNED_WHITE_KNIGHT = "4k3/8/8/8/4r3/4N3/8/4K3 w - - 0 1"
_KING_IN_CHECK = "4k3/8/8/4q3/8/8/8/4K3 w - - 0 1"
_ADVANCED_WHITE_PAWN = "4k3/8/8/8/3P4/8/8/4K3 w - - 0 1"
_OPENING_MIDDLE = "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 4"
_SEMI_OPEN_E_FILE = "rnbqkb1r/pppp1ppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"


def _moves_df(
    *,
    fen_before: str,
    fen_after: str,
    move_uci: str,
    opponent_move_was_capture: bool = False,
) -> pl.DataFrame:
    return pl.DataFrame({
        "fen_before": [fen_before],
        "fen_after": [fen_after],
        "move_uci": [move_uci],
        "opponent_move_was_capture": [opponent_move_was_capture],
    })


def _push(fen: str, move_uci: str) -> str:
    board = chess.Board(fen)
    board.push(chess.Move.from_uci(move_uci))
    return board.fen()


# --- Structural FEN metrics ---


def test_vertical_openness_all_open_with_kings_only() -> None:
    assert fen_vertical_openness(_KINGS_ONLY) == pytest.approx(8.0)


def test_vertical_openness_semi_open_e_file_after_e4() -> None:
    assert fen_vertical_openness(_SEMI_OPEN_E_FILE) == pytest.approx(0.5)


def test_vertical_openness_transformation_detects_change() -> None:
    result = VerticalOpennessTransformation().transform(
        _moves_df(fen_before=_START, fen_after=_KINGS_ONLY, move_uci="e2e4")
    )
    assert result["vertical_openness_before"][0] == pytest.approx(0.0)
    assert result["vertical_openness_after"][0] == pytest.approx(8.0)
    assert result["vertical_openness_delta"][0] == pytest.approx(8.0)


def test_diagonal_openness_all_open_with_kings_only() -> None:
    assert fen_diagonal_openness(_KINGS_ONLY) == pytest.approx(6.0)


def test_diagonal_openness_transformation_before_after_delta() -> None:
    result = DiagonalOpennessTransformation().transform(
        _moves_df(fen_before=_START, fen_after=_KINGS_ONLY, move_uci="e2e4")
    )
    assert result["diagonal_openness_before"][0] == pytest.approx(0.0)
    assert result["diagonal_openness_after"][0] == pytest.approx(6.0)


def test_pawn_tension_counts_adjacent_attacking_pairs() -> None:
    assert fen_pawn_tension(_START) == pytest.approx(0.0)
    assert fen_pawn_tension(_E4_D5) == pytest.approx(1.0)


def test_pawn_tension_transformation_delta() -> None:
    fen_after = _push(_START, "e2e4")
    result = PawnTensionTransformation().transform(
        _moves_df(fen_before=_START, fen_after=fen_after, move_uci="e2e4")
    )
    assert result["pawn_tension_before"][0] == pytest.approx(0.0)
    assert result["pawn_tension_after"][0] == pytest.approx(0.0)


# --- Dual-sided tactical metrics ---


def test_king_safety_starting_position_higher_than_exposed() -> None:
    board_start = chess.Board(_START)
    board_exposed = chess.Board(_KING_IN_CHECK)
    safe = fen_king_safety(board_start, chess.WHITE)
    exposed = fen_king_safety(board_exposed, chess.WHITE)
    assert safe > exposed
    assert safe == pytest.approx(8.25)


def test_king_safety_transformation_both_colors() -> None:
    result = KingSafetyTransformation().transform(
        _moves_df(fen_before=_START, fen_after=_KING_IN_CHECK, move_uci="e2e4")
    )
    assert result["white_king_safety_after"][0] > 0.0
    assert result["black_king_safety_after"][0] > 0.0


def test_mean_rank_increases_when_pawn_advances() -> None:
    board = chess.Board(_ADVANCED_WHITE_PAWN)
    assert fen_mean_rank(board, chess.WHITE) > fen_mean_rank(chess.Board(_START), chess.WHITE)


def test_mean_rank_transformation_delta_positive_for_pawn_push() -> None:
    fen_after = _push(_START, "e2e4")
    result = MeanRankTransformation().transform(
        _moves_df(fen_before=_START, fen_after=fen_after, move_uci="e2e4")
    )
    assert result["white_mean_rank_delta"][0] > 0.0


def test_attack_pressure_positive_when_knight_is_hanging() -> None:
    board = chess.Board(_HANGING_WHITE_KNIGHT)
    assert fen_attack_pressure(board, chess.WHITE) == pytest.approx(3.0)
    assert fen_attack_pressure(board, chess.BLACK) == pytest.approx(0.0)


def test_attack_pressure_transformation() -> None:
    result = AttackPressureTransformation().transform(
        _moves_df(
            fen_before=_HANGING_WHITE_KNIGHT,
            fen_after=_HANGING_WHITE_KNIGHT,
            move_uci="d5f4",
        )
    )
    assert result["white_attack_pressure_after"][0] == pytest.approx(3.0)


def test_hanging_value_counts_only_pawns_and_minors() -> None:
    board = chess.Board(_HANGING_WHITE_KNIGHT)
    assert fen_hanging_value(board, chess.WHITE) == pytest.approx(3.0)
    assert fen_has_hanging_piece(board, chess.WHITE) is True


def test_hanging_value_ignores_hanging_rook() -> None:
    # White rook on a8 attacked by black queen, undefended.
    fen = "4k3/8/8/8/8/7q/8/R3K3 w - - 0 1"
    board = chess.Board(fen)
    assert fen_hanging_value(board, chess.WHITE) == pytest.approx(0.0)
    assert fen_has_hanging_piece(board, chess.WHITE) is False


def test_hanging_value_transformation() -> None:
    result = HangingValueTransformation().transform(
        _moves_df(
            fen_before=_HANGING_WHITE_KNIGHT,
            fen_after=_HANGING_WHITE_KNIGHT,
            move_uci="d5f4",
        )
    )
    assert result["white_hanging_value_after"][0] == pytest.approx(3.0)


def test_pin_value_absolute_pin_to_king() -> None:
    board = chess.Board(_PINNED_WHITE_KNIGHT)
    assert board.is_pinned(chess.WHITE, chess.E3)
    assert fen_pin_value(board, chess.WHITE) == pytest.approx(3.75)


def test_pin_value_transformation() -> None:
    result = PinValueTransformation().transform(
        _moves_df(
            fen_before=_PINNED_WHITE_KNIGHT,
            fen_after=_PINNED_WHITE_KNIGHT,
            move_uci="e3g4",
        )
    )
    assert result["white_pin_value_after"][0] == pytest.approx(3.75)
    assert result["black_pin_value_after"][0] == pytest.approx(0.0)


# --- Move semantics ---


def test_move_flags_castle() -> None:
    move = chess.Move.from_uci("e1g1")
    assert move_is_castle(move) is True


def test_move_flags_gave_check() -> None:
    fen_before = "4k3/8/8/8/5Q2/8/8/4K3 w - - 0 1"
    board = chess.Board(fen_before)
    assert move_gave_check(board, chess.Move.from_uci("f4f7")) is True


def test_move_flags_promotion_and_en_passant() -> None:
    promo_move = chess.Move.from_uci("e7e8q")
    assert move_is_promotion(promo_move) is True

    ep_board = chess.Board("rnbqkbnr/ppp1ppp1/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
    ep_move = chess.Move.from_uci("e5d6")
    assert move_is_en_passant(ep_board, ep_move) is True


def test_move_flags_transformation_special_moves() -> None:
    fen_before = "8/4P3/8/8/8/8/8/4K2k w - - 0 1"
    fen_after = _push(fen_before, "e7e8q")
    result = MoveFlagsTransformation().transform(
        _moves_df(fen_before=fen_before, fen_after=fen_after, move_uci="e7e8q")
    )
    assert result["is_promotion"][0] is True
    assert result["is_capture"][0] is False


# --- Move context ---


def test_move_context_in_check_before() -> None:
    assert fen_is_in_check(_KING_IN_CHECK) is True
    result = MoveContextTransformation().transform(
        _moves_df(
            fen_before=_KING_IN_CHECK,
            fen_after=_KING_IN_CHECK,
            move_uci="e1e2",
        )
    )
    assert result["is_in_check_before"][0] is True


def test_move_context_hanging_pieces_before() -> None:
    result = MoveContextTransformation().transform(
        _moves_df(
            fen_before=_HANGING_WHITE_KNIGHT,
            fen_after=_HANGING_WHITE_KNIGHT,
            move_uci="d5f4",
        )
    )
    assert result["user_has_hanging_piece_before"][0] is True
    assert result["opponent_has_hanging_piece_before"][0] is False


def test_move_context_castling_rights() -> None:
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert fen_has_castling_rights(board, chess.WHITE) is True
    board_no_rights = chess.Board(_KINGS_ONLY)
    assert fen_has_castling_rights(board_no_rights, chess.WHITE) is False


def test_move_context_game_phase_mutually_exclusive() -> None:
    opening, middle, end = fen_game_phase(_OPENING_MIDDLE)
    assert opening is True
    assert middle is False
    assert end is False

    opening, middle, end = fen_game_phase(_KINGS_ONLY)
    assert opening is False
    assert middle is False
    assert end is True

    result = MoveContextTransformation().transform(
        _moves_df(fen_before=_OPENING_MIDDLE, fen_after=_OPENING_MIDDLE, move_uci="e2e4")
    )
    assert result["is_opening"][0] is True
    assert result["is_middle_game"][0] is False
    assert result["is_end_game"][0] is False


# --- Material (before column retained via metadata) ---


def test_material_balance_before_after_delta_consistent() -> None:
    from chess_teacher.pipelines.preprocessing.move_characteristics.material_balance import (
        MaterialBalanceTransformation,
    )

    fen_before = "rnbqkbnr/ppp1pppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    fen_after = _push(fen_before, "h2h4")
    result = MaterialBalanceTransformation().transform(
        _moves_df(fen_before=fen_before, fen_after=fen_after, move_uci="h2h4")
    )
    before = result["material_balance_before"][0]
    after = result["material_balance_after"][0]
    delta = result["material_balance_delta"][0]
    assert before == pytest.approx(1.0)
    assert after == pytest.approx(1.0)
    assert delta == pytest.approx(after - before)


# --- Legal moves (mobility, not complexity) ---


def test_legal_moves_black_unchanged_after_opening_e4() -> None:
    from chess_teacher.pipelines.preprocessing.move_characteristics.legal_moves import (
        LegalMovesTransformation,
    )

    fen_after = _push(_START, "e2e4")
    result = LegalMovesTransformation().transform(
        _moves_df(fen_before=_START, fen_after=fen_after, move_uci="e2e4")
    )
    assert result["white_legal_moves_after"][0] == pytest.approx(30.0)
    assert result["black_legal_moves_after"][0] == pytest.approx(20.0)
    assert result["black_legal_moves_delta"][0] == pytest.approx(0.0)
    assert result["white_legal_moves_delta"][0] == pytest.approx(10.0)
