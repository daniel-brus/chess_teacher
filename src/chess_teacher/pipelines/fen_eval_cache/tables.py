"""Position-eval cache table (Postgres schema ``engine``)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from chess_teacher.utils.table_data_class import TableDataClass


@dataclass
class PositionEvalRow(TableDataClass):
    epd: str
    engine: str
    eval_white_pov: float
    eval_depth: int
    candidates: dict[str, Any]
    candidate_nodes: int
    last_used: datetime

    @classmethod
    def get_yaml_path(cls) -> Path:
        return Path(__file__).parent / "metadata.yml"

    @classmethod
    def get_key(cls) -> str:
        return "position_evals"

    @classmethod
    def get_id_hash_columns(cls) -> tuple[str, ...]:
        return ("epd",)
