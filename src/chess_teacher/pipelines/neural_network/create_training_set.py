"""Convert DB move rows into ML training datums (first pass, imperfect OK)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Literal

import chess
import numpy as np

from chess_teacher.pipelines.neural_network.move_encoding import (
    POLICY_VOCAB_SIZE,
    MoveEncoder,
)
from chess_teacher.pipelines.preprocessing.games import Game
from chess_teacher.pipelines.preprocessing.moves import Move, MoveCharacteristics
from chess_teacher.utils.chess_utils import Color
from chess_teacher.utils.db.client import DatabaseClient, get_db_client
from chess_teacher.utils.general_utils import generate_ident_is_literal, quote_literal
from chess_teacher.utils.logging import get_logger

logger = get_logger()

# Shared FROM/WHERE for moves that have characteristics + candidate SF evals + known end_time.
_SQL_MOVES_WITH_CHARS = """
            FROM games.moves m
            INNER JOIN games.games g ON g.game_id = m.game_id
            INNER JOIN games.move_characteristics mc ON mc.move_id = m.move_id
            WHERE g.end_time IS NOT NULL
              AND mc.candidate_evaluations IS NOT NULL
"""

# Large move_characteristics JSONB + parallel hash join can OOM Postgres workers on
# memory-tight hosts (e.g. after a train batch). Keep these scans single-threaded.
_MOVES_QUERY_SESSION_SETTINGS = {"max_parallel_workers_per_gather": "0"}

# Domain scales tuned against local games.move_characteristics (n=825, dev_local).
# Absolute mate-like evals (~±100) intentionally saturate under tanh.
_EVAL_TANH_SCALE = 5.0  # typical |eval| p95 ~8-11; mates → ±1
_MATERIAL_TANH_SCALE = 15.0  # |material| max ~18 in sample
_LEGAL_MOVES_SCALE = 50.0  # after p99 ~53, delta |max| ~46
_KING_SAFETY_MAX = 8.25  # observed hard ceiling in sample
_KING_SAFETY_DELTA_SCALE = 4.0  # |delta| max ~4
_ATTACK_PRESSURE_TANH_SCALE = 8.0  # after p95 ~4-5, max 18 (was 20 → too flat)
_HANGING_VALUE_TANH_SCALE = 5.0  # after p95 ~3, max 12 (was 10 → too flat)
_PIN_VALUE_TANH_SCALE = 4.0  # after p95 ~1-1.25, rare spikes to ~13
_MEAN_RANK_MAX = 1.0
_VERTICAL_OPENNESS_MAX = 8.0
_DIAGONAL_OPENNESS_MAX = 6.0
_OPENNESS_DELTA_SCALE = 1.0  # |delta| max ≈ 1 in sample (do not use after-max)
_PAWN_TENSION_SCALE = 2.0  # after max 2 in sample (was 8 → crushed)
_BOARD_SPAN = 7.0  # files/ranks 0..7; deltas -7..7

_PIECE_NAME: dict[chess.PieceType, str] = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}

# (white_attr, black_attr, user_out, opponent_out) for side-symmetric metrics.
_SIDE_METRIC_PAIRS: tuple[tuple[str, str, str, str], ...] = (
    (
        "white_legal_moves_after",
        "black_legal_moves_after",
        "user_legal_moves_after",
        "opponent_legal_moves_after",
    ),
    (
        "white_legal_moves_delta",
        "black_legal_moves_delta",
        "user_legal_moves_delta",
        "opponent_legal_moves_delta",
    ),
    (
        "white_king_safety_after",
        "black_king_safety_after",
        "user_king_safety_after",
        "opponent_king_safety_after",
    ),
    (
        "white_king_safety_delta",
        "black_king_safety_delta",
        "user_king_safety_delta",
        "opponent_king_safety_delta",
    ),
    (
        "white_mean_rank_after",
        "black_mean_rank_after",
        "user_mean_rank_after",
        "opponent_mean_rank_after",
    ),
    (
        "white_mean_rank_delta",
        "black_mean_rank_delta",
        "user_mean_rank_delta",
        "opponent_mean_rank_delta",
    ),
    (
        "white_attack_pressure_after",
        "black_attack_pressure_after",
        "user_attack_pressure_after",
        "opponent_attack_pressure_after",
    ),
    (
        "white_attack_pressure_delta",
        "black_attack_pressure_delta",
        "user_attack_pressure_delta",
        "opponent_attack_pressure_delta",
    ),
    (
        "white_hanging_value_after",
        "black_hanging_value_after",
        "user_hanging_value_after",
        "opponent_hanging_value_after",
    ),
    (
        "white_hanging_value_delta",
        "black_hanging_value_delta",
        "user_hanging_value_delta",
        "opponent_hanging_value_delta",
    ),
    (
        "white_pin_value_after",
        "black_pin_value_after",
        "user_pin_value_after",
        "opponent_pin_value_after",
    ),
    (
        "white_pin_value_delta",
        "black_pin_value_delta",
        "user_pin_value_delta",
        "opponent_pin_value_delta",
    ),
)

_SIGNED_WHITE_POV_ATTRS: tuple[str, ...] = (
    "evaluation_before",
    "evaluation_after",
    "evaluation_delta",
    "material_balance_before",
    "material_balance_after",
    "material_balance_delta",
)

_PASSTHROUGH_NUMERIC_ATTRS: tuple[str, ...] = (
    "vertical_openness_after",
    "vertical_openness_delta",
    "diagonal_openness_after",
    "diagonal_openness_delta",
    "pawn_tension_after",
    "pawn_tension_delta",
)

_PASSTHROUGH_BOOL_ATTRS: tuple[str, ...] = (
    "is_capture",
    "is_castle",
    "gave_check",
    "created_fork",
    "is_promotion",
    "is_en_passant",
    "is_recapture",
    "is_in_check_before",
    "user_has_hanging_piece_before",
    "opponent_has_hanging_piece_before",
    "has_castling_rights_before",
    "is_opening",
    "is_middle_game",
    "is_end_game",
    "opponent_move_was_capture",
)

# Avoid SELECT * / PGN egress when hydrating training rows.
_GAME_TRAINING_COLUMNS: list[str] = ["game_id", "color"]
_MOVE_TRAINING_COLUMNS: list[str] = [
    "move_id",
    "game_id",
    "account_id",
    "move_nr",
    "ply",
    "move_san",
    "move_uci",
    "fen_before",
    "fen_after",
    "previous_opponent_move_uci",
]
_CHARS_TRAINING_COLUMNS: list[str] = list(
    dict.fromkeys([
        "move_id",
        "game_id",
        "account_id",
        *_SIGNED_WHITE_POV_ATTRS,
        *(white for white, _, _, _ in _SIDE_METRIC_PAIRS),
        *(black for _, black, _, _ in _SIDE_METRIC_PAIRS),
        *_PASSTHROUGH_NUMERIC_ATTRS,
        *_PASSTHROUGH_BOOL_ATTRS,
        "candidate_evaluations",
    ])
)

# Pre-move / context features only (no chosen-move leakage for move prediction).
_STATE_FEATURE_KEYS: tuple[str, ...] = (
    "evaluation_before_user_pov",
    "material_balance_before_user_pov",
    "is_in_check_before",
    "user_has_hanging_piece_before",
    "opponent_has_hanging_piece_before",
    "has_castling_rights_before",
    "is_opening",
    "is_middle_game",
    "is_end_game",
    "opponent_move_was_capture",
)

_PLY_NORM_SCALE = 80.0

_PIECE_TYPE_CATEGORIES: tuple[str, ...] = (
    "pawn",
    "knight",
    "bishop",
    "rook",
    "queen",
    "king",
    "none",
)

_PIECE_TYPE_TO_ONEHOT_IDX: dict[str, int] = {
    cat: idx for idx, cat in enumerate(_PIECE_TYPE_CATEGORIES)
}

# Feature-key → how to normalize continuous DB metrics (bools stay 0/1).
_FeatureNorm = Literal["bool", "unit01", "tanh", "div"]
_FEATURE_NORM_SPECS: dict[str, tuple[_FeatureNorm, float]] = {
    "evaluation_before_user_pov": ("tanh", _EVAL_TANH_SCALE),
    "evaluation_after_user_pov": ("tanh", _EVAL_TANH_SCALE),
    "evaluation_delta_user_pov": ("tanh", _EVAL_TANH_SCALE),
    "material_balance_before_user_pov": ("tanh", _MATERIAL_TANH_SCALE),
    "material_balance_after_user_pov": ("tanh", _MATERIAL_TANH_SCALE),
    "material_balance_delta_user_pov": ("tanh", _MATERIAL_TANH_SCALE),
    "user_legal_moves_after": ("div", _LEGAL_MOVES_SCALE),
    "opponent_legal_moves_after": ("div", _LEGAL_MOVES_SCALE),
    "user_legal_moves_delta": ("div", _LEGAL_MOVES_SCALE),
    "opponent_legal_moves_delta": ("div", _LEGAL_MOVES_SCALE),
    "user_king_safety_after": ("unit01", _KING_SAFETY_MAX),
    "opponent_king_safety_after": ("unit01", _KING_SAFETY_MAX),
    "user_king_safety_delta": ("div", _KING_SAFETY_DELTA_SCALE),
    "opponent_king_safety_delta": ("div", _KING_SAFETY_DELTA_SCALE),
    "user_mean_rank_after": ("unit01", _MEAN_RANK_MAX),
    "opponent_mean_rank_after": ("unit01", _MEAN_RANK_MAX),
    "user_mean_rank_delta": ("div", _MEAN_RANK_MAX),
    "opponent_mean_rank_delta": ("div", _MEAN_RANK_MAX),
    "user_attack_pressure_after": ("tanh", _ATTACK_PRESSURE_TANH_SCALE),
    "opponent_attack_pressure_after": ("tanh", _ATTACK_PRESSURE_TANH_SCALE),
    "user_attack_pressure_delta": ("tanh", _ATTACK_PRESSURE_TANH_SCALE),
    "opponent_attack_pressure_delta": ("tanh", _ATTACK_PRESSURE_TANH_SCALE),
    "user_hanging_value_after": ("tanh", _HANGING_VALUE_TANH_SCALE),
    "opponent_hanging_value_after": ("tanh", _HANGING_VALUE_TANH_SCALE),
    "user_hanging_value_delta": ("tanh", _HANGING_VALUE_TANH_SCALE),
    "opponent_hanging_value_delta": ("tanh", _HANGING_VALUE_TANH_SCALE),
    "user_pin_value_after": ("tanh", _PIN_VALUE_TANH_SCALE),
    "opponent_pin_value_after": ("tanh", _PIN_VALUE_TANH_SCALE),
    "user_pin_value_delta": ("tanh", _PIN_VALUE_TANH_SCALE),
    "opponent_pin_value_delta": ("tanh", _PIN_VALUE_TANH_SCALE),
    "vertical_openness_after": ("unit01", _VERTICAL_OPENNESS_MAX),
    "vertical_openness_delta": ("div", _OPENNESS_DELTA_SCALE),
    "diagonal_openness_after": ("unit01", _DIAGONAL_OPENNESS_MAX),
    "diagonal_openness_delta": ("div", _OPENNESS_DELTA_SCALE),
    "pawn_tension_after": ("div", _PAWN_TENSION_SCALE),
    "pawn_tension_delta": ("div", _PAWN_TENSION_SCALE),
}


def _float32_or_zero(value: Any) -> np.float32:
    if value is None:
        return np.float32(0.0)
    if isinstance(value, bool):
        return np.float32(1.0 if value else 0.0)
    return np.float32(value)


def _normalize_scalar(
    value: Any,
    *,
    kind: _FeatureNorm,
    scale: float,
) -> tuple[np.float32, np.float32]:
    """Return (normalized_value, missing_indicator). Missing → 0.0 + indicator 1.0."""
    if value is None:
        return np.float32(0.0), np.float32(1.0)
    if isinstance(value, bool):
        return np.float32(1.0 if value else 0.0), np.float32(0.0)

    x = float(value)
    if kind == "bool":
        y = 1.0 if x else 0.0
    elif kind == "unit01":
        y = min(max(x / scale, 0.0), 1.0) if scale else 0.0
    elif kind == "div":
        y = x / scale if scale else 0.0
    elif kind == "tanh":
        y = float(np.tanh(x / scale)) if scale else 0.0
    else:
        y = x
    return np.float32(y), np.float32(0.0)


def _one_hot_piece_type(piece_type: str | None) -> list[np.float32]:
    key = piece_type if piece_type is not None else "none"
    idx = _PIECE_TYPE_TO_ONEHOT_IDX.get(key, _PIECE_TYPE_TO_ONEHOT_IDX["none"])
    onehot = [np.float32(0.0)] * len(_PIECE_TYPE_CATEGORIES)
    onehot[idx] = np.float32(1.0)
    return onehot


def _piece_name(piece: chess.Piece | None) -> str | None:
    if piece is None:
        return None
    return _PIECE_NAME[piece.piece_type]


def piece_type_name(piece: chess.Piece | None) -> str | None:
    """Public alias of piece-type string used in training identity features."""
    return _piece_name(piece)


def _square_file_rank(square: chess.Square) -> tuple[int, int]:
    return chess.square_file(square), chess.square_rank(square)


def _flip_signed(value: float | None, *, flip: bool) -> float | None:
    if value is None:
        return None
    return -value if flip else value


class TrainingDatumBuilder:
    """Build ``TrainingDatum`` rows from DB moves / characteristics / games."""

    @staticmethod
    def derive_move_identity(
        *,
        fen_before: str,
        move_uci: str,
        previous_opponent_move_uci: str | None = None,
    ) -> dict[str, Any]:
        """Cheap geometry / piece-type features from FEN + UCI (not stored in DB)."""
        board = chess.Board(fen_before)
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            raise ValueError(f"Move {move_uci!r} illegal in fen_before={fen_before!r}")

        moved = board.piece_at(move.from_square)
        captured = board.piece_at(move.to_square)
        if board.is_en_passant(move):
            captured = chess.Piece(chess.PAWN, not board.turn)

        from_file, from_rank = _square_file_rank(move.from_square)
        to_file, to_rank = _square_file_rank(move.to_square)
        chebyshev = max(abs(to_file - from_file), abs(to_rank - from_rank))

        opponent_piece_type: str | None = None
        if previous_opponent_move_uci:
            try:
                opp = chess.Move.from_uci(previous_opponent_move_uci)
                opponent_piece_type = _piece_name(board.piece_at(opp.to_square))
            except ValueError:
                opponent_piece_type = None

        promotion_piece_type: str | None = None
        if move.promotion is not None:
            promotion_piece_type = _PIECE_NAME[move.promotion]

        return {
            "piece_type": _piece_name(moved),
            "captured_piece_type": _piece_name(captured),
            "promotion_piece_type": promotion_piece_type,
            "from_file": from_file,
            "from_rank": from_rank,
            "to_file": to_file,
            "to_rank": to_rank,
            "delta_file": to_file - from_file,
            "delta_rank": to_rank - from_rank,
            "move_distance_chebyshev": chebyshev,
            "is_kingside_castle": board.is_kingside_castling(move),
            "is_queenside_castle": board.is_queenside_castling(move),
            "legal_move_ucis": tuple(sorted(m.uci() for m in board.legal_moves)),
            "is_knight_move": moved is not None and moved.piece_type == chess.KNIGHT,
            "is_bishop_move": moved is not None and moved.piece_type == chess.BISHOP,
            "is_rook_move": moved is not None and moved.piece_type == chess.ROOK,
            "is_queen_move": moved is not None and moved.piece_type == chess.QUEEN,
            "is_king_move": moved is not None and moved.piece_type == chess.KING,
            "is_pawn_move": moved is not None and moved.piece_type == chess.PAWN,
            **opponent_piece_type_flags(opponent_piece_type),
        }

    @staticmethod
    def remap_characteristics_to_user_pov(
        chars: MoveCharacteristics,
        color: Color,
    ) -> dict[str, Any]:
        """Rewrite white/black metrics as user/opponent; sign-flip for Black."""
        flip = color == Color.BLACK
        out: dict[str, Any] = {}

        for attr in _SIGNED_WHITE_POV_ATTRS:
            out[f"{attr}_user_pov"] = _flip_signed(getattr(chars, attr), flip=flip)

        for white_attr, black_attr, user_key, opp_key in _SIDE_METRIC_PAIRS:
            white_val = getattr(chars, white_attr)
            black_val = getattr(chars, black_attr)
            if color == Color.WHITE:
                out[user_key] = white_val
                out[opp_key] = black_val
            else:
                out[user_key] = black_val
                out[opp_key] = white_val

        for attr in _PASSTHROUGH_NUMERIC_ATTRS:
            out[attr] = getattr(chars, attr)

        for attr in _PASSTHROUGH_BOOL_ATTRS:
            out[attr] = getattr(chars, attr)

        return out

    @classmethod
    def from_db_rows(
        cls,
        move: Move,
        chars: MoveCharacteristics,
        *,
        color: Color,
        game_id: str | None = None,
    ) -> TrainingDatum:
        """Pure convert: DB move + characteristics + game color → training datum.

        ``game_id`` defaults to ``move.game_id``. Pass explicitly when validating a
        joined game row without hydrating full ``Game`` (avoids PGN egress).
        """
        if move.move_id != chars.move_id:
            raise ValueError(f"move_id mismatch: move={move.move_id!r} chars={chars.move_id!r}")
        resolved_game_id = game_id if game_id is not None else move.game_id
        if move.game_id != resolved_game_id:
            raise ValueError(f"game_id mismatch: move={move.game_id!r} game={resolved_game_id!r}")

        identity = cls.derive_move_identity(
            fen_before=move.fen_before,
            move_uci=move.move_uci,
            previous_opponent_move_uci=move.previous_opponent_move_uci,
        )
        features = cls.remap_characteristics_to_user_pov(chars, color)

        return TrainingDatum(
            move_id=move.move_id,
            game_id=move.game_id,
            account_id=move.account_id,
            move_nr=move.move_nr,
            ply=move.ply,
            color=color,
            fen_before=move.fen_before,
            fen_after=move.fen_after,
            move_uci=move.move_uci,
            move_san=move.move_san,
            piece_type=identity["piece_type"],
            captured_piece_type=identity["captured_piece_type"],
            promotion_piece_type=identity["promotion_piece_type"],
            from_file=identity["from_file"],
            from_rank=identity["from_rank"],
            to_file=identity["to_file"],
            to_rank=identity["to_rank"],
            delta_file=identity["delta_file"],
            delta_rank=identity["delta_rank"],
            move_distance_chebyshev=identity["move_distance_chebyshev"],
            is_kingside_castle=identity["is_kingside_castle"],
            is_queenside_castle=identity["is_queenside_castle"],
            opponent_piece_type=identity["opponent_piece_type"],
            legal_move_ucis=identity["legal_move_ucis"],
            is_knight_move=identity["is_knight_move"],
            is_bishop_move=identity["is_bishop_move"],
            is_rook_move=identity["is_rook_move"],
            is_queen_move=identity["is_queen_move"],
            is_king_move=identity["is_king_move"],
            is_pawn_move=identity["is_pawn_move"],
            opponent_move_was_pawn=identity["opponent_move_was_pawn"],
            opponent_move_was_knight=identity["opponent_move_was_knight"],
            opponent_move_was_bishop=identity["opponent_move_was_bishop"],
            opponent_move_was_rook=identity["opponent_move_was_rook"],
            opponent_move_was_queen=identity["opponent_move_was_queen"],
            opponent_move_was_king=identity["opponent_move_was_king"],
            features=features,
            candidate_evaluations=chars.candidate_evaluations,
        )


def _append_feature_keys(
    out: list[np.float32],
    features: dict[str, Any],
    keys: tuple[str, ...] | list[str],
    *,
    normalize: bool,
    include_missing_indicators: bool,
) -> None:
    for key in keys:
        raw = features.get(key)
        if key in _PASSTHROUGH_BOOL_ATTRS:
            out.append(_float32_or_zero(raw))
            continue
        if normalize:
            kind, scale = _FEATURE_NORM_SPECS.get(key, ("div", 1.0))
            value, missing = _normalize_scalar(raw, kind=kind, scale=scale)
            out.append(value)
            if include_missing_indicators:
                out.append(missing)
        else:
            out.append(_float32_or_zero(raw))
            if include_missing_indicators:
                out.append(np.float32(1.0 if raw is None else 0.0))


def opponent_piece_type_flags(piece_type: str | None) -> dict[str, Any]:
    """Six opponent-piece one-hots + ``opponent_piece_type`` string (shared train/live)."""
    return {
        "opponent_piece_type": piece_type,
        "opponent_move_was_pawn": piece_type == "pawn",
        "opponent_move_was_knight": piece_type == "knight",
        "opponent_move_was_bishop": piece_type == "bishop",
        "opponent_move_was_rook": piece_type == "rook",
        "opponent_move_was_queen": piece_type == "queen",
        "opponent_move_was_king": piece_type == "king",
    }


def assemble_state_vector(
    *,
    opponent_move_was_pawn: bool,
    opponent_move_was_knight: bool,
    opponent_move_was_bishop: bool,
    opponent_move_was_rook: bool,
    opponent_move_was_queen: bool,
    opponent_move_was_king: bool,
    color_is_white: bool,
    ply: int,
    features: dict[str, Any],
    normalize: bool = True,
    include_missing_indicators: bool = True,
) -> np.ndarray:
    """Shared layout for ``TrainingDatum.state_vector`` and live board encoding."""
    opponent_piece_flags = [
        _float32_or_zero(opponent_move_was_pawn),
        _float32_or_zero(opponent_move_was_knight),
        _float32_or_zero(opponent_move_was_bishop),
        _float32_or_zero(opponent_move_was_rook),
        _float32_or_zero(opponent_move_was_queen),
        _float32_or_zero(opponent_move_was_king),
    ]
    color_flag = [_float32_or_zero(color_is_white)]
    if normalize:
        ply_part = [np.float32(min(ply / _PLY_NORM_SCALE, 1.0))]
    else:
        ply_part = [_float32_or_zero(ply)]

    feature_values: list[np.float32] = []
    _append_feature_keys(
        feature_values,
        features,
        _STATE_FEATURE_KEYS,
        normalize=normalize,
        include_missing_indicators=include_missing_indicators,
    )
    return np.asarray(
        opponent_piece_flags + color_flag + ply_part + feature_values,
        dtype=np.float32,
    )


@dataclass(frozen=True)
class TrainingDatum:
    """One labeled user-move example for move-choice / style models."""

    move_id: str
    game_id: str
    account_id: str
    move_nr: int
    ply: int
    color: Color

    fen_before: str
    fen_after: str
    move_uci: str
    move_san: str

    # Derived identity / geometry
    piece_type: str | None
    captured_piece_type: str | None
    promotion_piece_type: str | None
    from_file: int
    from_rank: int
    to_file: int
    to_rank: int
    delta_file: int
    delta_rank: int
    move_distance_chebyshev: int
    is_kingside_castle: bool
    is_queenside_castle: bool
    opponent_piece_type: str | None
    legal_move_ucis: tuple[str, ...]
    is_knight_move: bool
    is_bishop_move: bool
    is_rook_move: bool
    is_queen_move: bool
    is_king_move: bool
    is_pawn_move: bool
    opponent_move_was_pawn: bool
    opponent_move_was_knight: bool
    opponent_move_was_bishop: bool
    opponent_move_was_rook: bool
    opponent_move_was_queen: bool
    opponent_move_was_king: bool

    # User-POV + passthrough characteristics
    features: dict[str, Any]
    # Optional jsonb from move_characteristics (white-POV after-evals per legal UCI).
    candidate_evaluations: dict[str, Any] | None = None

    def state_vector(
        self,
        *,
        normalize: bool = True,
        include_missing_indicators: bool = True,
    ) -> np.ndarray:
        """Pre-move context features only (no chosen-move geometry / post-move metrics)."""
        return assemble_state_vector(
            opponent_move_was_pawn=self.opponent_move_was_pawn,
            opponent_move_was_knight=self.opponent_move_was_knight,
            opponent_move_was_bishop=self.opponent_move_was_bishop,
            opponent_move_was_rook=self.opponent_move_was_rook,
            opponent_move_was_queen=self.opponent_move_was_queen,
            opponent_move_was_king=self.opponent_move_was_king,
            color_is_white=self.color == Color.WHITE,
            ply=self.ply,
            features=self.features,
            normalize=normalize,
            include_missing_indicators=include_missing_indicators,
        )

    def action_label(self, *, normalize: bool = True) -> np.ndarray:
        """Legacy MSE target: board coords + moved-piece one-hot (unused by policy trainer)."""
        if normalize:
            coords = [
                np.float32(self.from_file / _BOARD_SPAN),
                np.float32(self.from_rank / _BOARD_SPAN),
                np.float32(self.to_file / _BOARD_SPAN),
                np.float32(self.to_rank / _BOARD_SPAN),
            ]
        else:
            coords = [
                _float32_or_zero(self.from_file),
                _float32_or_zero(self.from_rank),
                _float32_or_zero(self.to_file),
                _float32_or_zero(self.to_rank),
            ]
        return np.asarray(coords + _one_hot_piece_type(self.piece_type), dtype=np.float32)

    def policy_class_index(self) -> int:
        """Fixed-vocab index of the played move (user-style imitation target)."""
        return MoveEncoder.encode(self.move_uci)

    def policy_legal_mask(self) -> np.ndarray:
        """Boolean mask over the fixed move vocab for legal moves in ``fen_before``."""
        return MoveEncoder.mask_from_ucis(self.legal_move_ucis)

    def policy_target(self) -> tuple[int, np.ndarray]:
        """Legacy fixed-vocab target (unused by candidate-style trainer)."""
        return self.policy_class_index(), self.policy_legal_mask()

    def candidate_style_target(
        self,
    ) -> tuple[np.ndarray, np.ndarray, int] | None:
        """Padded candidate feats/mask/label, or None if evals missing/unusable."""
        from chess_teacher.pipelines.neural_network.candidate_eval import (
            pack_candidate_tensors,
            parse_candidate_evaluations,
        )

        payload = parse_candidate_evaluations(self.candidate_evaluations)
        if payload is None:
            return None
        evals = payload["evals_white_pov"]
        color_is_white = self.color == Color.WHITE
        eval_user = self.features.get("evaluation_before_user_pov")
        evaluation_before_white: float | None
        if eval_user is None:
            evaluation_before_white = None
        else:
            evaluation_before_white = float(eval_user) if color_is_white else float(-eval_user)
        return pack_candidate_tensors(
            evals,
            fen_before=self.fen_before,
            color_is_white=color_is_white,
            user_move_uci=self.move_uci,
            legal_ucis=self.legal_move_ucis,
            opponent_move_was_capture=bool(self.features.get("opponent_move_was_capture") or False),
            evaluation_before_white=evaluation_before_white,
        )

    def to_keras_input_vector(
        self,
        *,
        normalize: bool = True,
        include_missing_indicators: bool = True,
    ) -> np.ndarray:
        """Legacy full vector (state + action). Prefer ``state_vector`` for training."""
        moved_piece_flags = [
            _float32_or_zero(self.is_pawn_move),
            _float32_or_zero(self.is_knight_move),
            _float32_or_zero(self.is_bishop_move),
            _float32_or_zero(self.is_rook_move),
            _float32_or_zero(self.is_queen_move),
            _float32_or_zero(self.is_king_move),
        ]
        castling_flags = [
            _float32_or_zero(self.is_kingside_castle),
            _float32_or_zero(self.is_queenside_castle),
        ]

        if normalize:
            coords = [
                np.float32(self.from_file / _BOARD_SPAN),
                np.float32(self.from_rank / _BOARD_SPAN),
                np.float32(self.to_file / _BOARD_SPAN),
                np.float32(self.to_rank / _BOARD_SPAN),
                np.float32(self.delta_file / _BOARD_SPAN),
                np.float32(self.delta_rank / _BOARD_SPAN),
                np.float32(self.move_distance_chebyshev / _BOARD_SPAN),
            ]
        else:
            coords = [
                _float32_or_zero(self.from_file),
                _float32_or_zero(self.from_rank),
                _float32_or_zero(self.to_file),
                _float32_or_zero(self.to_rank),
                _float32_or_zero(self.delta_file),
                _float32_or_zero(self.delta_rank),
                _float32_or_zero(self.move_distance_chebyshev),
            ]

        captured_onehot = _one_hot_piece_type(self.captured_piece_type)
        promotion_onehot = _one_hot_piece_type(self.promotion_piece_type)

        opponent_piece_flags = [
            _float32_or_zero(self.opponent_move_was_pawn),
            _float32_or_zero(self.opponent_move_was_knight),
            _float32_or_zero(self.opponent_move_was_bishop),
            _float32_or_zero(self.opponent_move_was_rook),
            _float32_or_zero(self.opponent_move_was_queen),
            _float32_or_zero(self.opponent_move_was_king),
        ]

        color_flag = [_float32_or_zero(self.color == Color.WHITE)]

        feature_values: list[np.float32] = []
        _append_feature_keys(
            feature_values,
            self.features,
            training_datum_feature_keys(),
            normalize=normalize,
            include_missing_indicators=include_missing_indicators,
        )

        vec = (
            moved_piece_flags
            + castling_flags
            + coords
            + captured_onehot
            + promotion_onehot
            + opponent_piece_flags
            + color_flag
            + feature_values
        )
        return np.asarray(vec, dtype=np.float32)

    def to_keras_inputs(
        self,
        *,
        normalize: bool = True,
        include_missing_indicators: bool = True,
        scaler: FeatureNormalizer | None = None,
    ) -> dict[str, np.ndarray]:
        """Return Keras-friendly inputs: state ``x`` and action ``y``."""
        x = self.state_vector(
            normalize=normalize,
            include_missing_indicators=include_missing_indicators,
        )
        if scaler is not None:
            x = scaler.transform(x)
        y = self.action_label(normalize=normalize)
        return {"x": x[None, :], "y": y[None, :]}

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["color"] = self.color.value
        return payload


@dataclass
class FeatureNormalizer:
    """Optional second-stage z-score fit on training vectors (needs a batch).

    Use after domain normalization: fit on train only, then transform train/val/test.
    """

    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    eps: float = 1e-6

    def fit(self, vectors: list[np.ndarray] | np.ndarray) -> FeatureNormalizer:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        self.mean = matrix.mean(axis=0).astype(np.float32)
        self.std = matrix.std(axis=0).astype(np.float32)
        self.std = np.where(self.std < self.eps, 1.0, self.std).astype(np.float32)
        return self

    def transform(self, vector: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("FeatureNormalizer.fit(...) before transform().")
        x = np.asarray(vector, dtype=np.float32)
        return ((x - self.mean) / self.std).astype(np.float32)

    def fit_transform(self, vectors: list[np.ndarray] | np.ndarray) -> np.ndarray:
        matrix = np.asarray(vectors, dtype=np.float32)
        self.fit(matrix)
        return self.transform_batch(matrix)

    def transform_batch(self, matrix: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("FeatureNormalizer.fit(...) before transform_batch().")
        x = np.asarray(matrix, dtype=np.float32)
        return ((x - self.mean) / self.std).astype(np.float32)


@dataclass(frozen=True)
class TrainingBatch:
    """A list of ``TrainingDatum`` with matrix helpers for Keras."""

    datums: list[TrainingDatum]

    def state_matrix(
        self,
        *,
        normalize: bool = True,
        include_missing_indicators: bool = True,
        scaler: FeatureNormalizer | None = None,
    ) -> np.ndarray:
        if not self.datums:
            return np.zeros((0, 0), dtype=np.float32)
        matrix = np.stack(
            [
                d.state_vector(
                    normalize=normalize,
                    include_missing_indicators=include_missing_indicators,
                )
                for d in self.datums
            ],
            axis=0,
        )
        if scaler is not None:
            matrix = scaler.transform_batch(matrix)
        return matrix.astype(np.float32)

    def action_matrix(self, *, normalize: bool = True) -> np.ndarray:
        if not self.datums:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack(
            [d.action_label(normalize=normalize) for d in self.datums],
            axis=0,
        ).astype(np.float32)

    def policy_class_indices(self) -> np.ndarray:
        if not self.datums:
            return np.zeros((0,), dtype=np.int32)
        return np.asarray(
            [d.policy_class_index() for d in self.datums],
            dtype=np.int32,
        )

    def policy_legal_masks(self) -> np.ndarray:
        if not self.datums:
            return np.zeros((0, POLICY_VOCAB_SIZE), dtype=np.bool_)
        return np.stack([d.policy_legal_mask() for d in self.datums], axis=0)

    def policy_targets(self) -> tuple[np.ndarray, np.ndarray]:
        """Legacy ``(y_index shape (N,), legal_mask shape (N, V))``."""
        return self.policy_class_indices(), self.policy_legal_masks()

    def candidate_style_targets(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
        """Stack candidate tensors; drop datums that cannot form a target.

        Returns ``(feats, mask, labels, kept_indices)`` where kept_indices map
        into ``self.datums``.
        """
        from chess_teacher.pipelines.neural_network.candidate_eval import (
            MAX_CANDIDATES,
            MOVE_FEAT_DIM,
        )

        n = len(self.datums)
        logger.info(
            "Packing candidate-style targets for %s datums (feat_dim=%s, max_candidates=%s)…",
            n,
            MOVE_FEAT_DIM,
            MAX_CANDIDATES,
        )
        feats_list: list[np.ndarray] = []
        mask_list: list[np.ndarray] = []
        labels: list[int] = []
        kept: list[int] = []
        progress_every = max(1, n // 5) if n else 1
        for i, d in enumerate(self.datums):
            packed = d.candidate_style_target()
            if packed is None:
                continue
            f, m, lab = packed
            feats_list.append(f)
            mask_list.append(m)
            labels.append(lab)
            kept.append(i)
            done = i + 1
            if done == n or (done % progress_every == 0):
                logger.info(
                    "Candidate feature progress %s/%s kept=%s",
                    done,
                    n,
                    len(kept),
                )
        if not feats_list:
            return (
                np.zeros((0, MAX_CANDIDATES, MOVE_FEAT_DIM), dtype=np.float32),
                np.zeros((0, MAX_CANDIDATES), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
                [],
            )
        return (
            np.stack(feats_list, axis=0),
            np.stack(mask_list, axis=0),
            np.asarray(labels, dtype=np.int32),
            kept,
        )

    def legacy_matrix(
        self,
        *,
        normalize: bool = True,
        include_missing_indicators: bool = True,
        scaler: FeatureNormalizer | None = None,
    ) -> np.ndarray:
        """Stack legacy full vectors (state + action). Prefer ``state_matrix``."""
        if not self.datums:
            return np.zeros((0, 0), dtype=np.float32)
        matrix = np.stack(
            [
                d.to_keras_input_vector(
                    normalize=normalize,
                    include_missing_indicators=include_missing_indicators,
                )
                for d in self.datums
            ],
            axis=0,
        )
        if scaler is not None:
            matrix = scaler.transform_batch(matrix)
        return matrix.astype(np.float32)


class TrainingDataStore:
    """Load training datums from Postgres (platform-wide or per account)."""

    def __init__(self, db_client: DatabaseClient | None = None) -> None:
        self._db = db_client or get_db_client()

    def _ensure_training_tables(self) -> None:
        self._db.ensure_tables(
            Move.get_metadata(),
            Game.get_metadata(),
            MoveCharacteristics.get_metadata(),
        )

    def _query_moves_sql(
        self,
        sql: str,
        params: dict[str, Any] | list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return self._db.engine.execute_parameterized_query(
            sql,
            params,
            session_settings=_MOVES_QUERY_SESSION_SETTINGS,
        )

    def _datums_from_moves(self, moves: list[Move]) -> list[TrainingDatum]:
        if not moves:
            return []
        self._ensure_training_tables()
        game_ids = sorted({m.game_id for m in moves})
        move_ids = [m.move_id for m in moves]
        move_id_list = ", ".join(quote_literal(mid) for mid in move_ids)
        game_id_list = ", ".join(quote_literal(gid) for gid in game_ids)

        chars_by_id = {
            c.move_id: c
            for c in MoveCharacteristics.fetch_all_from_db(
                self._db,
                columns=_CHARS_TRAINING_COLUMNS,
                where=f'"move_id" IN ({move_id_list})',
            )
        }
        color_by_game_id = {
            row["game_id"]: Color(row["color"])
            for row in self._db.read(
                Game.get_metadata(),
                columns=_GAME_TRAINING_COLUMNS,
                where=f'"game_id" IN ({game_id_list})',
            )
        }

        datums: list[TrainingDatum] = []
        for move in moves:
            chars = chars_by_id.get(move.move_id)
            color = color_by_game_id.get(move.game_id)
            if chars is None or color is None:
                continue
            try:
                datums.append(
                    TrainingDatumBuilder.from_db_rows(
                        move,
                        chars,
                        color=color,
                        game_id=move.game_id,
                    )
                )
            except ValueError:
                continue
        return datums

    def _datums_for_move_ids(self, move_ids: list[str]) -> list[TrainingDatum]:
        """Hydrate ordered ``move_id`` list into datums (missing ids skipped)."""
        if not move_ids:
            return []
        logger.info(
            "Hydrating %s moves into TrainingDatum (moves + characteristics + games)…",
            len(move_ids),
        )
        move_id_list = ", ".join(quote_literal(mid) for mid in move_ids)
        moves = Move.fetch_all_from_db(
            self._db,
            columns=_MOVE_TRAINING_COLUMNS,
            where=f'"move_id" IN ({move_id_list})',
        )
        by_id = {m.move_id: m for m in moves}
        ordered_moves = [by_id[mid] for mid in move_ids if mid in by_id]
        return self._datums_from_moves(ordered_moves)

    def fetch_one(self, move_id: str) -> TrainingDatum:
        move = Move.fetch_from_db(self._db, id=move_id, columns=_MOVE_TRAINING_COLUMNS)
        chars = MoveCharacteristics.fetch_from_db(
            self._db, id=move_id, columns=_CHARS_TRAINING_COLUMNS
        )
        game_rows = self._db.read(
            Game.get_metadata(),
            columns=_GAME_TRAINING_COLUMNS,
            where=generate_ident_is_literal("game_id", move.game_id),
        )
        if len(game_rows) != 1:
            raise ValueError(
                f"Expected one game for game_id={move.game_id!r}, got {len(game_rows)}"
            )
        return TrainingDatumBuilder.from_db_rows(
            move,
            chars,
            color=Color(game_rows[0]["color"]),
            game_id=str(game_rows[0]["game_id"]),
        )

    def fetch_for_account(
        self,
        account_id: str,
        *,
        limit: int | None = None,
    ) -> list[TrainingDatum]:
        moves = Move.fetch_all_from_db(
            self._db,
            columns=_MOVE_TRAINING_COLUMNS,
            where=generate_ident_is_literal("account_id", account_id),
            order_by='"ply" ASC',
            limit=limit,
        )
        return self._datums_from_moves(moves)

    def count_since(self, cutoff: datetime | None) -> int:
        """Count platform moves with characteristics and ``games.end_time`` after cutoff."""
        self._ensure_training_tables()
        sql = f"SELECT COUNT(*) AS n{_SQL_MOVES_WITH_CHARS}"
        params: dict[str, Any] = {}
        if cutoff is not None:
            sql += " AND g.end_time > :cutoff"
            params["cutoff"] = cutoff
        rows = self._query_moves_sql(sql, params)
        return int(rows[0]["n"]) if rows else 0

    def fetch_since(
        self,
        cutoff: datetime | None,
        *,
        limit: int | None = None,
    ) -> tuple[list[TrainingDatum], datetime | None]:
        """Load new rows ordered by ``games.end_time`` (oldest first).

        Returns ``(datums, max_end_time)``.

        When ``limit`` truncates mid-``end_time`` group (all moves in a game share
        ``games.end_time``), the batch is expanded to include **every** move at
        that boundary timestamp so the next cutoff ``end_time > max`` cannot skip
        the rest of the game / same-second games.
        """
        self._ensure_training_tables()
        sql = f"SELECT m.move_id AS move_id, g.end_time AS end_time{_SQL_MOVES_WITH_CHARS}"
        params: dict[str, Any] = {}
        if cutoff is not None:
            sql += " AND g.end_time > :cutoff"
            params["cutoff"] = cutoff
        sql += " ORDER BY g.end_time ASC, m.game_id ASC, m.move_nr ASC"
        if limit is not None:
            sql += " LIMIT :limit"
            params["limit"] = limit

        logger.info(
            "Querying training move ids (cutoff=%s limit=%s)…",
            cutoff,
            limit,
        )
        rows = self._query_moves_sql(sql, params)
        if not rows:
            return [], None

        # LIMIT may cut inside a shared end_time group — finish that group.
        if limit is not None and len(rows) >= limit:
            max_end_time = max(r["end_time"] for r in rows if r["end_time"] is not None)
            prefix = [r for r in rows if r["end_time"] is not None and r["end_time"] < max_end_time]
            expand_sql = (
                f"SELECT m.move_id AS move_id, g.end_time AS end_time{_SQL_MOVES_WITH_CHARS}"
                " AND g.end_time = :boundary"
                " ORDER BY m.game_id ASC, m.move_nr ASC"
            )
            at_boundary = self._query_moves_sql(expand_sql, {"boundary": max_end_time})
            rows = prefix + list(at_boundary)

        move_ids = [str(r["move_id"]) for r in rows]
        end_times = [r["end_time"] for r in rows if r["end_time"] is not None]
        max_end_time = max(end_times) if end_times else None
        return self._datums_for_move_ids(move_ids), max_end_time

    def fetch_random(
        self,
        *,
        limit: int,
        seed: int | None = None,
    ) -> list[TrainingDatum]:
        """Sample up to ``limit`` moves with characteristics.

        With ``seed``, ordering is deterministic via ``md5(move_id || seed)``
        (safe with pooled connections — unlike session ``setseed``).
        Without ``seed``, uses ``ORDER BY random()``.
        """
        if limit <= 0:
            return []
        self._ensure_training_tables()
        params: dict[str, Any] = {"limit": limit}
        if seed is not None:
            order = "ORDER BY md5(m.move_id || :seed_text)"
            params["seed_text"] = str(seed)
        else:
            order = "ORDER BY random()"
        sql = f"SELECT m.move_id AS move_id{_SQL_MOVES_WITH_CHARS} {order} LIMIT :limit"
        rows = self._query_moves_sql(sql, params)
        if not rows:
            return []
        move_ids = [str(r["move_id"]) for r in rows]
        return self._datums_for_move_ids(move_ids)


def training_datum_feature_keys() -> list[str]:
    """Stable list of keys inside TrainingDatum.features (for debugging schemas)."""
    dummy_keys = (
        [f"{a}_user_pov" for a in _SIGNED_WHITE_POV_ATTRS]
        + [user for _, _, user, _ in _SIDE_METRIC_PAIRS]
        + [opp for _, _, _, opp in _SIDE_METRIC_PAIRS]
        + list(_PASSTHROUGH_NUMERIC_ATTRS)
        + list(_PASSTHROUGH_BOOL_ATTRS)
    )
    return dummy_keys


def training_datum_state_feature_keys() -> list[str]:
    """Stable list of pre-move feature keys used by ``state_vector``."""
    return list(_STATE_FEATURE_KEYS)


# Backward-compatible aliases
derive_move_identity = TrainingDatumBuilder.derive_move_identity
remap_characteristics_to_user_pov = TrainingDatumBuilder.remap_characteristics_to_user_pov
move_in_database_to_training_datum = TrainingDatumBuilder.from_db_rows


def stack_keras_batch(
    datums: list[TrainingDatum],
    *,
    normalize: bool = True,
    include_missing_indicators: bool = True,
    scaler: FeatureNormalizer | None = None,
) -> np.ndarray:
    return TrainingBatch(datums).legacy_matrix(
        normalize=normalize,
        include_missing_indicators=include_missing_indicators,
        scaler=scaler,
    )


def stack_state_batch(
    datums: list[TrainingDatum],
    *,
    normalize: bool = True,
    include_missing_indicators: bool = True,
    scaler: FeatureNormalizer | None = None,
) -> np.ndarray:
    return TrainingBatch(datums).state_matrix(
        normalize=normalize,
        include_missing_indicators=include_missing_indicators,
        scaler=scaler,
    )


def stack_action_labels(
    datums: list[TrainingDatum],
    *,
    normalize: bool = True,
) -> np.ndarray:
    return TrainingBatch(datums).action_matrix(normalize=normalize)


def fetch_training_datum(
    move_id: str,
    *,
    db_client: DatabaseClient | None = None,
) -> TrainingDatum:
    return TrainingDataStore(db_client).fetch_one(move_id)


def fetch_training_data_for_account(
    account_id: str,
    *,
    db_client: DatabaseClient | None = None,
    limit: int | None = None,
) -> list[TrainingDatum]:
    return TrainingDataStore(db_client).fetch_for_account(account_id, limit=limit)


def count_new_moves_since(
    cutoff: datetime | None,
    *,
    db_client: DatabaseClient | None = None,
) -> int:
    return TrainingDataStore(db_client).count_since(cutoff)


def fetch_training_data_since(
    cutoff: datetime | None,
    *,
    db_client: DatabaseClient | None = None,
    limit: int | None = None,
) -> tuple[list[TrainingDatum], datetime | None]:
    return TrainingDataStore(db_client).fetch_since(cutoff, limit=limit)


__all__ = [
    "FeatureNormalizer",
    "TrainingBatch",
    "TrainingDataStore",
    "TrainingDatum",
    "TrainingDatumBuilder",
    "count_new_moves_since",
    "derive_move_identity",
    "fetch_training_data_for_account",
    "fetch_training_data_since",
    "fetch_training_datum",
    "move_in_database_to_training_datum",
    "remap_characteristics_to_user_pov",
    "stack_action_labels",
    "stack_keras_batch",
    "stack_state_batch",
    "training_datum_feature_keys",
    "training_datum_state_feature_keys",
]
