"""Local file helpers and structured file loaders/writers.

Prefer importing from ``file_utils``, ``file_loader``, or ``file_writer`` directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chess_teacher.utils.files.file_loader import (
        FileLoader,
        FileLoaderFactory,
        JsonlLoader,
        TsvLoader,
    )
    from chess_teacher.utils.files.file_utils import FileType, TextStreamSource
    from chess_teacher.utils.files.file_writer import FileWriter, FileWriterFactory, JsonlWriter

__all__ = [
    "FileLoader",
    "FileLoaderFactory",
    "FileType",
    "FileWriter",
    "FileWriterFactory",
    "JsonlLoader",
    "JsonlWriter",
    "TextStreamSource",
    "TsvLoader",
    "check_destination_for_write",
    "copy_file",
    "discover_files",
    "ensure_destination_parent",
    "move_file",
    "remove_file",
    "validate_existing_file",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "FileLoader": (".file_loader", "FileLoader"),
    "FileLoaderFactory": (".file_loader", "FileLoaderFactory"),
    "JsonlLoader": (".file_loader", "JsonlLoader"),
    "TsvLoader": (".file_loader", "TsvLoader"),
    "FileType": (".file_utils", "FileType"),
    "TextStreamSource": (".file_utils", "TextStreamSource"),
    "check_destination_for_write": (".file_utils", "check_destination_for_write"),
    "copy_file": (".file_utils", "copy_file"),
    "discover_files": (".file_utils", "discover_files"),
    "ensure_destination_parent": (".file_utils", "ensure_destination_parent"),
    "move_file": (".file_utils", "move_file"),
    "remove_file": (".file_utils", "remove_file"),
    "validate_existing_file": (".file_utils", "validate_existing_file"),
    "FileWriter": (".file_writer", "FileWriter"),
    "FileWriterFactory": (".file_writer", "FileWriterFactory"),
    "JsonlWriter": (".file_writer", "JsonlWriter"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        import importlib

        module_name, attr = _LAZY_ATTRS[name]
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
