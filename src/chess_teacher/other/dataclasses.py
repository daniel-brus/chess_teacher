from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from chess_teacher.utils.table_data_class import TableDataClass


# NOTE: Classical is excluded for now since it is not handled correctly by ingestion.
class TimeControlCategory(StrEnum):
    ULTRA_BULLET = "UltraBullet"
    BULLET = "Bullet"
    BLITZ = "Blitz"
    RAPID = "Rapid"
    UNKNOWN = "Unknown"

    @classmethod
    def from_initial_and_increment(
        cls,
        initial: int | None,
        increment: int | None = None,
    ) -> TimeControlCategory:
        if initial is None:
            return cls.UNKNOWN
        estimate = initial + (increment or 0) * 40
        if estimate < 30:
            return cls.ULTRA_BULLET
        if estimate < 180:
            return cls.BULLET
        if estimate < 600:
            return cls.BLITZ
        if estimate >= 600:
            return cls.RAPID
        return cls.UNKNOWN


@dataclass
class TimeControlClass(TableDataClass):
    """Represents a time control class."""

    time_control_class_id: str
    time_control_class: TimeControlCategory

    @classmethod
    def get_yaml_path(cls) -> Path:
        return Path(__file__).parent / "metadata.yml"

    @classmethod
    def get_key(cls) -> str:
        return "time_control_classes"

    @classmethod
    def get_id_hash_columns(cls) -> tuple[str, ...]:
        return ("time_control_class",)

    @classmethod
    def from_initial_and_increment(
        cls,
        initial: int | None,
        increment: int | None = None,
    ) -> TimeControlClass:
        category = TimeControlCategory.from_initial_and_increment(initial, increment)
        return cls(
            time_control_class_id=cls.generate_id({"time_control_class": category.value}),
            time_control_class=category,
        )
