from __future__ import annotations

from collections.abc import Sequence
from typing import Literal
from uuid import uuid4

from chess_teacher.pipelines.ingestion.raw_games import RawGame
from chess_teacher.pipelines.preprocessing.games import Game
from chess_teacher.pipelines.preprocessing.moves import Move
from chess_teacher.platform.account import Account
from chess_teacher.platform.user import User
from chess_teacher.utils.db.client import DatabaseClient
from chess_teacher.utils.exception_utils import FileError
from chess_teacher.utils.general_utils import quote_ident, quote_literal
from chess_teacher.utils.logging import EnhancedLogger, get_logger
from chess_teacher.utils.object_storage.base import ObjectStorage
from chess_teacher.utils.object_storage.factory import get_raw_storage

_SOURCE_FOLDERS: tuple[Literal["processed", "failed"], ...] = ("processed", "failed")
_INGESTED_FOLDER: Literal["ingested"] = "ingested"


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
    accounts = Account.fetch_all_from_db(db_client)
    log.info(f"Backfill move starting for {len(accounts)} account(s)")

    for account in accounts:
        move_processed_and_failed_to_ingested(
            account.account_id,
            overwrite=overwrite,
            storage=storage,
            logger=log,
        )

    log.info(f"Backfill move finished for {len(accounts)} account(s)")


def _where_account_ids(account_ids: Sequence[str]) -> str:
    if not account_ids:
        return "FALSE"
    in_list = ", ".join(quote_literal(account_id) for account_id in account_ids)
    return f"{quote_ident('account_id')} IN ({in_list})"


def clear_user_game_tables(
    db_client: DatabaseClient,
    user_id: str,
    *,
    logger: EnhancedLogger | None = None,
) -> None:
    """
    Delete all ``raw_games``, ``games``, and ``moves`` rows for the user's linked accounts.

    Accounts with no rows are skipped without error.
    """
    log = logger or get_logger()
    user = User.fetch_from_db(db_client, id=user_id)
    accounts = user.get_linked_accounts(db_client)
    account_ids = [account.account_id for account in accounts]

    if not account_ids:
        log.info(f"No linked accounts for user {user_id}; nothing to clear")
        return

    where = _where_account_ids(account_ids)
    deleted_total = 0
    for data_class in (Move, Game, RawGame):
        table = data_class.get_metadata()
        db_client.ensure_metadata(table)
        deleted = db_client.delete_where(table, where=where)
        log.info(f"Cleared {deleted} row(s) from {table.qualified_name_sql()} for user {user_id}")
        deleted_total += deleted

    log.info(
        f"Cleared {deleted_total} game-table row(s) across {len(account_ids)} account(s) "
        f"for user {user_id}"
    )
