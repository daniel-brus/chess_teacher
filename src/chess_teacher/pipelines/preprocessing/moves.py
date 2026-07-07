from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from chess_teacher.utils.table_data_class import TableDataClass


@dataclass(frozen=True)
class Move(TableDataClass):
    move_id: str
    game_id: str
    account_id: str
    move_nr: int
    ply: int
    move_san: str
    move_uci: str
    fen_before: str
    fen_after: str
    previous_opponent_move_san: str | None = None
    previous_opponent_move_uci: str | None = None
    opponent_move_was_capture: bool = False

    @classmethod
    def get_yaml_path(cls) -> Path:
        return Path(__file__).parent / "metadata.yml"

    @classmethod
    def get_key(cls) -> str:
        return "moves"

    @classmethod
    def get_id_hash_columns(cls) -> tuple[str, ...]:
        return ("game_id", "move_nr")


@dataclass(frozen=True)
class MoveCharacteristics(TableDataClass):
    move_id: str
    game_id: str
    account_id: str
    evaluation_after: float | None = None
    evaluation_before: float | None = None
    evaluation_delta: float | None = None
    material_balance_after: float | None = None
    material_balance_before: float | None = None
    material_balance_delta: float | None = None
    vertical_openness_after: float | None = None
    vertical_openness_delta: float | None = None
    diagonal_openness_after: float | None = None
    diagonal_openness_delta: float | None = None
    pawn_tension_after: float | None = None
    pawn_tension_delta: float | None = None
    white_legal_moves_after: float | None = None
    white_legal_moves_delta: float | None = None
    black_legal_moves_after: float | None = None
    black_legal_moves_delta: float | None = None
    white_king_safety_after: float | None = None
    white_king_safety_delta: float | None = None
    black_king_safety_after: float | None = None
    black_king_safety_delta: float | None = None
    white_mean_rank_after: float | None = None
    white_mean_rank_delta: float | None = None
    black_mean_rank_after: float | None = None
    black_mean_rank_delta: float | None = None
    white_attack_pressure_after: float | None = None
    white_attack_pressure_delta: float | None = None
    black_attack_pressure_after: float | None = None
    black_attack_pressure_delta: float | None = None
    white_hanging_value_after: float | None = None
    white_hanging_value_delta: float | None = None
    black_hanging_value_after: float | None = None
    black_hanging_value_delta: float | None = None
    white_pin_value_after: float | None = None
    white_pin_value_delta: float | None = None
    black_pin_value_after: float | None = None
    black_pin_value_delta: float | None = None
    is_capture: bool = False
    is_castle: bool = False
    gave_check: bool = False
    created_fork: bool = False
    is_promotion: bool = False
    is_en_passant: bool = False
    is_recapture: bool = False
    is_in_check_before: bool = False
    user_has_hanging_piece_before: bool = False
    opponent_has_hanging_piece_before: bool = False
    has_castling_rights_before: bool = False
    is_opening: bool = False
    is_middle_game: bool = False
    is_end_game: bool = False
    previous_opponent_move_san: str | None = None
    previous_opponent_move_uci: str | None = None
    opponent_move_was_capture: bool = False

    @classmethod
    def get_yaml_path(cls) -> Path:
        return Path(__file__).parent / "metadata.yml"

    @classmethod
    def get_key(cls) -> str:
        return "move_characteristics"

    @classmethod
    def get_id_hash_columns(cls) -> tuple[str, ...]:
        return ()
