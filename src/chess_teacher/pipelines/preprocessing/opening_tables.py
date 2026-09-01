"""Opening reference tables (Postgres schema ``other``)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from chess_teacher.utils.table_data_class import TableDataClass


@dataclass
class RawEcoCode(TableDataClass):
    """Raw ECO opening row from lichess chess-openings TSV files."""

    eco_code_id: str
    eco_code: str
    name: str
    pgn: str

    @classmethod
    def get_yaml_path(cls) -> Path:
        return Path(__file__).parent / "metadata.yml"

    @classmethod
    def get_key(cls) -> str:
        return "raw_eco_codes"

    @classmethod
    def get_id_hash_columns(cls) -> tuple[str, ...]:
        return ("pgn",)


@dataclass
class RawChessComOpening(TableDataClass):
    """Chess.com opening slug and display title."""

    chess_com_opening_id: str
    slug: str
    name: str

    @classmethod
    def get_yaml_path(cls) -> Path:
        return Path(__file__).parent / "metadata.yml"

    @classmethod
    def get_key(cls) -> str:
        return "raw_chess_com_openings"

    @classmethod
    def get_id_hash_columns(cls) -> tuple[str, ...]:
        return ("slug",)

    @classmethod
    def from_slug_and_name(cls, slug: str, name: str) -> RawChessComOpening:
        return cls(
            chess_com_opening_id=cls.generate_id({"slug": slug}),
            slug=slug,
            name=name,
        )
