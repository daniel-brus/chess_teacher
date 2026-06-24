from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from chess_teacher.utils.chess_utils import Color, Reason, Result
from chess_teacher.utils.table_data_class import TableDataClass


@dataclass(frozen=True)
class Game(TableDataClass):
    game_id: str
    platform_game_id: str
    account_id: str
    raw_pgn: str
    cleaned_pgn: str
    color: Color
    result: Result
    reason: Reason
    time_control_initial: int | None = None
    time_control_increment: int | None = None
    variant: str = "standard"
    start_time: datetime | None = None
    end_time: datetime | None = None
    eco_code: str | None = None
    chess_com_opening_slug: str | None = None
    user_elo: int | None = None
    opponent_elo: int | None = None
    opening_name: str | None = None
    opening_family: str | None = None

    @classmethod
    def get_yaml_path(cls) -> Path:
        return Path(__file__).parent / "metadata.yml"

    @classmethod
    def get_key(cls) -> str:
        return "games"

    @classmethod
    def get_id_hash_columns(cls) -> tuple[str, ...]:
        return ("account_id", "platform_game_id")
