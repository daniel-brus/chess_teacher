from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

from chess_teacher.platform.account import Account
from chess_teacher.utils.db_client import DatabaseClient
from chess_teacher.utils.env_utils import get_env_variable
from chess_teacher.utils.exception_utils import ConfigError, FileError
from chess_teacher.utils.file_utils import discover_files, move_file
from chess_teacher.utils.logging_utils import EnhancedLogger, get_logger

_SOURCE_FOLDERS: tuple[Literal["processed", "failed"], ...] = ("processed", "failed")
_INGESTED_FOLDER: Literal["ingested"] = "ingested"


def fetch_all_accounts(db_client: DatabaseClient) -> list[Account]:
    """Load every row from the accounts table."""
    rows = cast(list[dict[str, Any]], db_client.read(Account.get_metadata()))
    return [Account.from_dict(row) for row in rows]


def _raw_dir() -> Path:
    try:
        raw_dir = get_env_variable("RAW_DIR")
    except ValueError as e:
        raise ConfigError(f"RAW_DIR environment variable is not set: {e}") from e
    if not raw_dir:
        raise ConfigError("RAW_DIR environment variable is not set")
    return Path(raw_dir)


def _account_folder_root(
    folder: Literal["ingested", "failed", "processed"],
    account_id: str,
    *,
    mkdir: bool,
) -> Path:
    path = _raw_dir() / folder / account_id
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path


def _destination_under_ingested(
    source: Path,
    source_root: Path,
    ingested_root: Path,
    *,
    overwrite: bool,
) -> Path:
    """Map a file under source_root to the same relative path under ingested_root."""
    destination = ingested_root / source.relative_to(source_root)
    if overwrite or not destination.exists():
        return destination
    return destination.with_name(f"{destination.stem}_{uuid4().hex}{destination.suffix}")


def _remove_empty_subdirectories(root: Path) -> None:
    """Remove empty subdirectories under root (deepest first). Keeps root itself."""
    if not root.is_dir():
        return
    for directory in sorted(
        (p for p in root.rglob("*") if p.is_dir()),
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            pass


def move_processed_and_failed_to_ingested(
    account_id: str,
    *,
    overwrite: bool = False,
    cleanup_empty_dirs: bool = True,
    logger: EnhancedLogger | None = None,
) -> None:
    """
    Move all files under ``{RAW_DIR}/processed/{account_id}`` and
    ``{RAW_DIR}/failed/{account_id}`` into ``{RAW_DIR}/ingested/{account_id}``,
    preserving each file's path relative to its source account folder.

    Only the top-level folder name changes (processed or failed → ingested);
    date subpaths (``YYYY/MM/DD/...``) and file names are unchanged.
    """
    log = logger or get_logger()
    ingested_root = _account_folder_root(_INGESTED_FOLDER, account_id, mkdir=True)
    moved_count = 0

    for folder in _SOURCE_FOLDERS:
        source_root = _account_folder_root(folder, account_id, mkdir=False)
        if not source_root.exists():
            log.info(f"No {folder} folder for account {account_id}: {source_root}")
            continue

        paths = discover_files(source_root, recursive=True, logger=log)
        if not paths:
            log.info(f"No files under {source_root}")
            continue

        for source in paths:
            destination = _destination_under_ingested(
                source,
                source_root,
                ingested_root,
                overwrite=overwrite,
            )
            try:
                move_file(
                    source,
                    destination,
                    overwrite=overwrite,
                    mkdir=True,
                    logger=log,
                )
            except FileError as e:
                raise FileError(f"Backfill move failed for {source} -> {destination}: {e}") from e
            log.info(f"Backfill moved {source} -> {destination}")
            moved_count += 1

        if cleanup_empty_dirs:
            _remove_empty_subdirectories(source_root)

    if moved_count:
        log.info(f"Backfill moved {moved_count} file(s) for account {account_id}")


def move_processed_and_failed_to_ingested_for_all_accounts(
    db_client: DatabaseClient,
    *,
    overwrite: bool = False,
    cleanup_empty_dirs: bool = True,
    logger: EnhancedLogger | None = None,
) -> None:
    """
    Run :func:`move_processed_and_failed_to_ingested` for every account in the DB.

    Accounts with no processed/failed folders or no files under them are skipped
    without error (see the single-account function).
    """
    log = logger or get_logger()
    accounts = fetch_all_accounts(db_client)
    log.info(f"Backfill move starting for {len(accounts)} account(s)")

    for account in accounts:
        move_processed_and_failed_to_ingested(
            account.account_id,
            overwrite=overwrite,
            cleanup_empty_dirs=cleanup_empty_dirs,
            logger=log,
        )

    log.info(f"Backfill move finished for {len(accounts)} account(s)")
