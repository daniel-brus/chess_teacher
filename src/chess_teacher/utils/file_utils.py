from __future__ import annotations

import re
import shutil
from enum import StrEnum
from pathlib import Path

from chess_teacher.utils.exception_utils import FileError
from chess_teacher.utils.logging_utils import EnhancedLogger, get_logger


class FileType(StrEnum):
    JSONL = "jsonl"
    # TODO: Add other file types


def validate_existing_file(
    path: Path,
    *,
    label: str = "File",
    logger: EnhancedLogger | None = None,
    error_type: type[FileError] = FileError,
) -> None:
    """Validate that path exists and refers to a regular file."""
    log = logger or get_logger()
    path = Path(path)

    if not path.name:
        log.log_and_raise(error_type(f"{label} path has no file name: {path}"))
    if not path.exists():
        log.log_and_raise(error_type(f"{label} file does not exist: {path}"))
    if path.is_dir():
        log.log_and_raise(error_type(f"{label} path is a directory: {path}"))
    if not path.is_file():
        log.log_and_raise(error_type(f"{label} path is not a file: {path}"))


def ensure_destination_parent(
    destination: Path,
    *,
    mkdir: bool = True,
    logger: EnhancedLogger | None = None,
    error_type: type[FileError] = FileError,
) -> None:
    """Validate the destination file name and ensure its parent directory exists."""
    log = logger or get_logger()
    destination = Path(destination)

    if not destination.name:
        log.log_and_raise(error_type(f"Destination path has no file name: {destination}"))

    parent = destination.parent

    if parent.exists():
        if not parent.is_dir():
            log.log_and_raise(error_type(f"Destination parent is not a directory: {parent}"))
    elif mkdir:
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            log.log_and_raise(error_type(f"Could not create destination parent {parent}: {e}"))
    else:
        log.log_and_raise(error_type(f"Destination parent directory does not exist: {parent}"))


def check_destination_for_write(
    destination: Path,
    *,
    overwrite: bool = False,
    logger: EnhancedLogger | None = None,
    error_type: type[FileError] = FileError,
) -> None:
    """Validate that an existing destination may be written (overwrite policy)."""
    log = logger or get_logger()
    destination = Path(destination)

    if not destination.exists():
        return
    if destination.is_dir():
        log.log_and_raise(error_type(f"Destination path is a directory: {destination}"))
    if not overwrite:
        log.log_and_raise(error_type(f"Destination file already exists: {destination}"))


def discover_files(
    path: Path,
    *,
    recursive: bool,
    suffix: str | None = None,
    glob_pattern: str | None = None,
    logger: EnhancedLogger | None = None,
) -> list[Path]:
    """
    Discover files to load under a storage path.

    - recursive=False: path must be a single existing file.
    - recursive=True: path must be a directory; returns all matching files under it
      (including subdirectories), sorted for deterministic loading.
    - suffix: when set (e.g. ``"jsonl"`` or ``".jsonl"``), only files with that
      suffix are included; when omitted, all regular files are candidates.
    """
    log = logger or get_logger()
    path = Path(path)
    normalized_suffix = suffix if suffix is None or suffix.startswith(".") else f".{suffix}"

    def _matches_suffix(file_path: Path) -> bool:
        if normalized_suffix is None:
            return True
        return file_path.suffix == normalized_suffix

    if recursive:
        if not path.exists():
            log.log_and_raise(FileError(f"Storage path does not exist: {path}"))
        if not path.is_dir():
            log.log_and_raise(
                FileError(
                    f"recursive=True requires a directory, but path is not a directory: {path}"
                )
            )
        files = sorted(p for p in path.rglob("*") if p.is_file() and _matches_suffix(p))
    else:
        validate_existing_file(path, logger=log, error_type=FileError)
        if not _matches_suffix(path):
            log.log_and_raise(
                FileError(f"Expected file with suffix {normalized_suffix}, got: {path}")
            )
        files = [path]

    if glob_pattern is not None:
        try:
            pattern = re.compile(glob_pattern)
        except re.error as e:
            log.log_and_raise(FileError(f"Invalid glob_pattern regex ({glob_pattern!r}): {e}"))

        try:
            files = [file_path for file_path in files if pattern.search(file_path.as_posix())]
        except Exception as e:
            log.log_and_raise(FileError(f"Error matching glob_pattern against files: {e}"))
    return files


def remove_file(
    path: Path,
    *,
    missing_ok: bool = False,
    logger: EnhancedLogger | None = None,
    error_type: type[FileError] = FileError,
) -> None:
    """Remove a single file. Does not remove directories."""
    log = logger or get_logger()
    path = Path(path)

    if not path.name:
        log.log_and_raise(error_type(f"File path has no file name: {path}"))

    if not path.exists():
        if missing_ok:
            return
        log.log_and_raise(error_type(f"File does not exist: {path}"))

    if path.is_dir():
        log.log_and_raise(error_type(f"Path is a directory, not a file: {path}"))

    try:
        path.unlink()
    except OSError as e:
        log.log_and_raise(error_type(f"Could not remove {path}: {e}"))


def move_file(
    source: Path,
    destination: Path,
    *,
    overwrite: bool = False,
    mkdir: bool = True,
    logger: EnhancedLogger | None = None,
    error_type: type[FileError] = FileError,
) -> None:
    """
    Move a file to a new path.

    Uses atomic replace when source and destination share a filesystem; falls back
    to shutil.move across devices. Ideal for promoting a completed temp file to
    its final path (e.g. data.jsonl.tmp -> data.jsonl).
    """
    log = logger or get_logger()
    source = Path(source)
    destination = Path(destination)

    if source == destination:
        log.log_and_raise(error_type(f"Source and destination are the same path: {source}"))

    validate_existing_file(source, label="Source", logger=log, error_type=error_type)
    ensure_destination_parent(destination, mkdir=mkdir, logger=log, error_type=error_type)
    check_destination_for_write(destination, overwrite=overwrite, logger=log, error_type=error_type)

    try:
        source.replace(destination)
    except OSError:
        try:
            shutil.move(source, destination)
        except OSError as e:
            log.log_and_raise(error_type(f"Could not move {source} to {destination}: {e}"))


def copy_file(
    source: Path,
    destination: Path,
    *,
    overwrite: bool = False,
    mkdir: bool = True,
    logger: EnhancedLogger | None = None,
    error_type: type[FileError] = FileError,
) -> None:
    """
    Copy a file to a new path.

    Copies to a temporary file next to the destination, then moves it into place
    so the destination either does not exist or is replaced atomically.
    """
    log = logger or get_logger()
    source = Path(source)
    destination = Path(destination)

    if source == destination:
        log.log_and_raise(error_type(f"Source and destination are the same path: {source}"))

    validate_existing_file(source, label="Source", logger=log, error_type=error_type)
    ensure_destination_parent(destination, mkdir=mkdir, logger=log, error_type=error_type)
    check_destination_for_write(destination, overwrite=overwrite, logger=log, error_type=error_type)

    tmp_path = destination.with_name(destination.name + ".tmp")
    try:
        shutil.copy2(source, tmp_path)
        move_file(
            tmp_path,
            destination,
            overwrite=True,
            mkdir=False,
            logger=log,
            error_type=error_type,
        )
    except FileError:
        remove_file(tmp_path, missing_ok=True, logger=log, error_type=error_type)
        raise
