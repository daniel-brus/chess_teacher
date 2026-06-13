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

    @classmethod
    def get_yaml_path(cls) -> Path:
        return Path(__file__).parent / "metadata.yml"

    @classmethod
    def get_key(cls) -> str:
        return "moves"

    @classmethod
    def get_id_hash_columns(cls) -> tuple[str, ...]:
        return ("game_id", "move_nr")
