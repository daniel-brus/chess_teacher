from __future__ import annotations

from typing import Any, Literal, cast
from uuid import uuid4

from chess_teacher.platform.account import Account
from chess_teacher.utils.db_client import DatabaseClient
from chess_teacher.utils.exception_utils import FileError
from chess_teacher.utils.logging_utils import EnhancedLogger, get_logger
from chess_teacher.utils.object_storage.base import ObjectStorage
from chess_teacher.utils.object_storage.factory import get_raw_storage

_SOURCE_FOLDERS: tuple[Literal["processed", "failed"], ...] = ("processed", "failed")
_INGESTED_FOLDER: Literal["ingested"] = "ingested"


def fetch_all_accounts(db_client: DatabaseClient) -> list[Account]:
    """Load every row from the accounts table."""
    rows = cast(list[dict[str, Any]], db_client.read(Account.get_metadata()))
    return [Account.from_dict(row) for row in rows]


def _account_folder_prefix(
    folder: Literal["ingested", "failed", "processed"], account_id: str
) -> str:
    return ObjectStorage.resolve_key(folder, account_id)


def _destination_under_ingested(
    source_key: str,
    source_prefix: str,
    ingested_prefix: str,
    *,
    overwrite: bool,
    storage: ObjectStorage,
) -> str:
    """Map an object under source_prefix to the same relative key under ingested_prefix."""
    relative = ObjectStorage.relative_key_under(source_key, source_prefix)
    destination = ObjectStorage.resolve_key(ingested_prefix, relative)
    if overwrite or storage.read_bytes(destination) is None:
        return destination
    return ObjectStorage.resolve_key(
        ingested_prefix,
        ObjectStorage.unique_key_variant(relative, uuid4().hex),
    )


def move_processed_and_failed_to_ingested(
    account_id: str,
    *,
    overwrite: bool = False,
    storage: ObjectStorage | None = None,
    logger: EnhancedLogger | None = None,
) -> None:
    """
    Move all objects under ``processed/{account_id}`` and ``failed/{account_id}``
    into ``ingested/{account_id}``, preserving each object's path relative to its
    source account prefix.
    """
    log = logger or get_logger()
    store = storage if storage is not None else get_raw_storage()
    ingested_prefix = _account_folder_prefix(_INGESTED_FOLDER, account_id)
    moved_count = 0

    for folder in _SOURCE_FOLDERS:
        source_prefix = _account_folder_prefix(folder, account_id)
        keys = store.list_keys(source_prefix, recursive=True)
        if not keys:
            log.info(f"No objects under {source_prefix}")
            continue

        for source_key in keys:
            destination = _destination_under_ingested(
                source_key,
                source_prefix,
                ingested_prefix,
                overwrite=overwrite,
                storage=store,
            )
            try:
                store.move(source_key, destination, overwrite=overwrite)
            except FileError as e:
                raise FileError(
                    f"Backfill move failed for {source_key} -> {destination}: {e}"
                ) from e
            log.info(f"Backfill moved {source_key} -> {destination}")
            moved_count += 1

    if moved_count:
        log.info(f"Backfill moved {moved_count} object(s) for account {account_id}")


def move_processed_and_failed_to_ingested_for_all_accounts(
    db_client: DatabaseClient,
    *,
    overwrite: bool = False,
    storage: ObjectStorage | None = None,
    logger: EnhancedLogger | None = None,
) -> None:
    """
    Run :func:`move_processed_and_failed_to_ingested` for every account in the DB.

    Accounts with no processed/failed prefixes or no objects under them are skipped
    without error (see the single-account function).
    """
    log = logger or get_logger()
    accounts = fetch_all_accounts(db_client)
    log.info(f"Backfill move starting for {len(accounts)} account(s)")

    for account in accounts:
        move_processed_and_failed_to_ingested(
            account.account_id,
            overwrite=overwrite,
            storage=storage,
            logger=log,
        )

    log.info(f"Backfill move finished for {len(accounts)} account(s)")


def archive_ingested_to_processed(
    account_id: str,
    *,
    storage: ObjectStorage | None = None,
    logger: EnhancedLogger | None = None,
) -> int:
    """Move all ``.jsonl`` objects from ``ingested/{account_id}`` to ``processed/{account_id}``.

    Uses :meth:`ObjectStorage.move_verified` and force-deletes any keys that still remain.
    Returns the number of objects archived. Safe to run when data is already in the DB.
    """
    log = logger or get_logger()
    store = storage if storage is not None else get_raw_storage()
    source_prefix = _account_folder_prefix("ingested", account_id)
    archive_prefix = _account_folder_prefix("processed", account_id)

    keys = store.list_keys(source_prefix, recursive=True, suffix="jsonl")
    if not keys:
        log.info(f"No ingested objects to archive under {source_prefix}")
        return 0

    archived = 0
    for source_key in keys:
        relative = ObjectStorage.relative_key_under(source_key, source_prefix)
        destination = ObjectStorage.resolve_key(archive_prefix, relative)
        if store.read_bytes(destination) is not None:
            destination = ObjectStorage.resolve_key(
                archive_prefix,
                ObjectStorage.unique_key_variant(relative, uuid4().hex),
            )
        store.move_verified(source_key, destination, overwrite=False)
        log.info(f"Backfill archived {source_key} -> {destination}")
        archived += 1

    leftover = store.list_keys(source_prefix, recursive=True, suffix="jsonl")
    if leftover:
        log.warning(
            f"{len(leftover)} object(s) still under {source_prefix} after archive; forcing delete."
        )
        store.delete_keys(leftover, missing_ok=False)

    log.info(f"Backfill archived {archived} object(s) for account {account_id}")
    return archived
