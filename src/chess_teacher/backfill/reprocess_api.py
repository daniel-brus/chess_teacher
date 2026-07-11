from __future__ import annotations

from typing import Literal

from chess_teacher.platform.account import Account
from chess_teacher.utils.db.client import DatabaseClient
from chess_teacher.utils.logging import EnhancedLogger, get_logger
from chess_teacher.utils.object_storage.base import ObjectStorage
from chess_teacher.utils.object_storage.factory import get_raw_storage

_ACCOUNT_STORAGE_FOLDERS: tuple[Literal["ingested", "failed", "processed"], ...] = (
    "ingested",
    "failed",
    "processed",
)


def _account_folder_prefix(
    folder: Literal["ingested", "failed", "processed"], account_id: str
) -> str:
    return ObjectStorage.resolve_key(folder, account_id)


def reset_account_latest_ingestion(
    db_client: DatabaseClient,
    account_id: str,
    *,
    logger: EnhancedLogger | None = None,
) -> None:
    """Clear ``latest_ingestion`` so the next API ingest fetches from the beginning."""
    log = logger or get_logger()
    account = Account.fetch_from_db(db_client, id=account_id)
    account.upsert_field(db_client, "latest_ingestion", None)
    log.info(f"Reset latest_ingestion for account {account_id}")


def clear_account_storage_folders(
    account_id: str,
    *,
    storage: ObjectStorage | None = None,
    logger: EnhancedLogger | None = None,
) -> None:
    """Delete all objects under ``ingested``, ``failed``, and ``processed`` for an account."""
    log = logger or get_logger()
    store = storage if storage is not None else get_raw_storage()
    deleted_count = 0

    for folder in _ACCOUNT_STORAGE_FOLDERS:
        prefix = _account_folder_prefix(folder, account_id)
        keys = store.list_keys(prefix, recursive=True)
        if not keys:
            log.info(f"No objects under {prefix}")
            continue
        store.delete_keys(keys)
        log.info(f"Deleted {len(keys)} object(s) under {prefix}")
        deleted_count += len(keys)

    if deleted_count:
        log.info(f"Cleared {deleted_count} storage object(s) for account {account_id}")


def reset_account_for_reprocess(
    db_client: DatabaseClient,
    account_id: str,
    *,
    storage: ObjectStorage | None = None,
    logger: EnhancedLogger | None = None,
) -> None:
    """
    Prepare one account for a full API re-ingest: reset ``latest_ingestion`` and
    wipe all account storage folders.
    """
    log = logger or get_logger()
    reset_account_latest_ingestion(db_client, account_id, logger=log)
    clear_account_storage_folders(account_id, storage=storage, logger=log)
    log.info(f"Account {account_id} prepared for reprocess")
