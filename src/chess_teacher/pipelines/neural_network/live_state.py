"""Live board → same ``state_vector`` layout used in baseline training."""

from __future__ import annotations

import chess
import numpy as np

from chess_teacher.pipelines.neural_network.create_training_set import (
    assemble_state_vector,
    opponent_piece_type_flags,
    piece_type_name,
)
from chess_teacher.utils.chess_utils import StockfishEngine, material_balance_white_pov
from chess_teacher.utils.chess_utils.fen_metrics import (
    fen_game_phase,
    fen_has_castling_rights,
    fen_has_hanging_piece,
)
from chess_teacher.utils.logging import get_logger

logger = get_logger()


def _opponent_move_flags(
    board: chess.Board,
    last_opponent_move_uci: str | None,
) -> dict[str, object]:
    """Infer opponent piece-type flags + capture flag from last move."""
    empty: dict[str, object] = {
        **opponent_piece_type_flags(None),
        "opponent_move_was_capture": False,
    }
    uci = last_opponent_move_uci
    if not uci and board.move_stack:
        uci = board.peek().uci()
    if not uci:
        return empty

    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        return empty

    # Prefer move_stack so capture/piece identity match training derive path.
    if board.move_stack and board.peek().uci() == uci:
        probe = board.copy(stack=True)
        last = probe.pop()
        was_capture = probe.is_capture(last)
        piece_type = piece_type_name(probe.piece_at(last.from_square))
    else:
        # Fallback: piece now sits on to-square; capture unknown → False.
        piece_type = piece_type_name(board.piece_at(move.to_square))
        was_capture = False

    return {
        **opponent_piece_type_flags(piece_type),
        "opponent_move_was_capture": was_capture,
    }


def compose_live_state_vector(
    board: chess.Board,
    *,
    evaluation_white_pov: float | None,
    last_opponent_move_uci: str | None = None,
    normalize: bool = True,
    include_missing_indicators: bool = True,
) -> np.ndarray:
    """Build training-compatible state vector; side-to-move = user POV."""
    user_color = board.turn
    flip = user_color == chess.BLACK
    fen = board.fen(en_passant="fen")

    material_white = material_balance_white_pov(fen)
    eval_user = (
        None
        if evaluation_white_pov is None
        else (-evaluation_white_pov if flip else evaluation_white_pov)
    )
    material_user = -material_white if flip else material_white

    is_opening, is_middle, is_end = fen_game_phase(fen)
    opp_flags = _opponent_move_flags(board, last_opponent_move_uci)

    features: dict[str, object] = {
        "evaluation_before_user_pov": eval_user,
        "material_balance_before_user_pov": material_user,
        "is_in_check_before": board.is_check(),
        "user_has_hanging_piece_before": fen_has_hanging_piece(board, user_color),
        "opponent_has_hanging_piece_before": fen_has_hanging_piece(board, not user_color),
        "has_castling_rights_before": fen_has_castling_rights(board, user_color),
        "is_opening": is_opening,
        "is_middle_game": is_middle,
        "is_end_game": is_end,
        "opponent_move_was_capture": bool(opp_flags["opponent_move_was_capture"]),
    }

    return assemble_state_vector(
        opponent_move_was_pawn=bool(opp_flags["opponent_move_was_pawn"]),
        opponent_move_was_knight=bool(opp_flags["opponent_move_was_knight"]),
        opponent_move_was_bishop=bool(opp_flags["opponent_move_was_bishop"]),
        opponent_move_was_rook=bool(opp_flags["opponent_move_was_rook"]),
        opponent_move_was_queen=bool(opp_flags["opponent_move_was_queen"]),
        opponent_move_was_king=bool(opp_flags["opponent_move_was_king"]),
        color_is_white=user_color == chess.WHITE,
        ply=board.ply(),
        features=features,
        normalize=normalize,
        include_missing_indicators=include_missing_indicators,
    )


_COMPUTE_EVAL = object()


class LiveStateEncoder:
    """Encode a live ``chess.Board`` with on-the-fly Stockfish eval (user decision)."""

    DEFAULT_STOCKFISH_DEPTH = 12

    def __init__(
        self,
        *,
        stockfish_depth: int = DEFAULT_STOCKFISH_DEPTH,
        stockfish_path: str | None = None,
        engine: StockfishEngine | None = None,
    ) -> None:
        self._owns_engine = engine is None
        self._engine = engine or StockfishEngine(
            depth=stockfish_depth,
            path=stockfish_path,
        )
        if self._owns_engine:
            self._engine.__enter__()

    def encode(
        self,
        board: chess.Board,
        *,
        last_opponent_move_uci: str | None = None,
        evaluation_white_pov: float | object | None = _COMPUTE_EVAL,
    ) -> np.ndarray:
        """Build state vector.

        Pass ``evaluation_white_pov`` to skip a second Stockfish search (e.g. reuse
        MultiPV root approx). Omit / leave default to compute via engine.
        """
        fen = board.fen(en_passant="fen")
        if evaluation_white_pov is _COMPUTE_EVAL:
            evaluation = self._engine.evaluate_white_pov_pawns(fen)
            if evaluation is None:
                logger.warning("Live Stockfish eval failed for fen=%s; missing indicator set", fen)
        else:
            evaluation = evaluation_white_pov  # type: ignore[assignment]
        return compose_live_state_vector(
            board,
            evaluation_white_pov=evaluation,
            last_opponent_move_uci=last_opponent_move_uci,
        )

    def close(self) -> None:
        if self._owns_engine:
            self._engine.__exit__(None, None, None)
            self._owns_engine = False

    def __enter__(self) -> LiveStateEncoder:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
