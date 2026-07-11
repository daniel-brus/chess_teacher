"""Local file helpers and structured file loaders/writers."""

from chess_teacher.utils.files.file_loader import (
    FileLoader,
    FileLoaderFactory,
    JsonlLoader,
    TsvLoader,
)
from chess_teacher.utils.files.file_utils import (
    FileType,
    TextStreamSource,
    check_destination_for_write,
    copy_file,
    discover_files,
    ensure_destination_parent,
    move_file,
    remove_file,
    validate_existing_file,
)
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
