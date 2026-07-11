from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from chess_teacher.utils.table_data_class import TableDataClass


@dataclass()
class UserAccount(TableDataClass):
    """Bridge row linking a user to a platform account."""

    user_id: str
    account_id: str

    @classmethod
    def get_key(cls) -> str:
        return "br_users_accounts"

    @classmethod
    def get_yaml_path(cls) -> Path:
        return Path(__file__).parent / "metadata.yml"

    @classmethod
    def get_id_hash_columns(cls) -> tuple[str, ...]:
        return ()
