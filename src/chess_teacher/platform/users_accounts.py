from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from chess_teacher.platform.account import Account
from chess_teacher.platform.user import User
from chess_teacher.utils.db_client import DatabaseClient
from chess_teacher.utils.exception_utils import DatabaseError
from chess_teacher.utils.general_utils import generate_ident_is_literal
from chess_teacher.utils.logging_utils import get_logger
from chess_teacher.utils.table_data_class import TableDataClass

logger = get_logger()


@dataclass()
class UserAccount(TableDataClass):
    """Represents a bridge between a user and an account."""

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


def get_accounts_for_user(user: User, db_client: DatabaseClient) -> list[Account]:
    """Fetch all platform accounts linked to a user."""

    db_client.ensure_table(Account.get_metadata())

    br_metadata = UserAccount.get_metadata()
    db_client.ensure_table(br_metadata)

    user_accounts = db_client.read(
        br_metadata,
        where=generate_ident_is_literal("user_id", user.user_id),
        order_by="account_id",
    )
    accounts: list[Account] = []
    for user_account in user_accounts:
        try:
            accounts.append(Account.fetch_from_db(db_client, id=user_account["account_id"]))
        except DatabaseError:
            logger.warning(
                "Removing stale account link for user %s and account %s",
                user.user_id,
                user_account["account_id"],
            )
            UserAccount.from_dict(user_account).delete_from_db(db_client)
    return accounts


def remove_all_accounts_for_user(user: User, db_client: DatabaseClient) -> None:
    """Remove all accounts from a user by removing the entries in the bridge table."""
    br_metadata = UserAccount.get_metadata()
    db_client.ensure_table(br_metadata)

    user_accounts = db_client.read(
        br_metadata,
        where=generate_ident_is_literal("user_id", user.user_id),
    )
    for user_account in user_accounts:
        UserAccount.from_dict(user_account).delete_from_db(db_client)


def add_account(user: User, account: Account, db_client: DatabaseClient) -> bool:
    """Add an account to the user. If the account already exists, return False."""
    account.save_new_to_db(db_client)
    user_account = UserAccount(
        user_id=user.user_id,
        account_id=account.account_id,
    )
    return user_account.save_new_to_db(db_client)


def remove_account(user: User, account: Account, db_client: DatabaseClient) -> bool:
    """Remove an account from the user by removing the entry in the bridge table."""
    user_account = UserAccount(
        user_id=user.user_id,
        account_id=account.account_id,
    )
    if not db_client.exists(UserAccount.get_metadata(), user_account.get_where_clause()):
        logger.log_and_raise(
            DatabaseError(f"Account {account.account_id} not found for user {user.user_id}")
        )
    user_account.delete_from_db(db_client)
    return True
