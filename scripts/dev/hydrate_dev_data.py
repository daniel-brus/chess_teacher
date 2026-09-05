"""Hydrate the local dev database with real, public chess games.

Creates a single dev user, links one or more platform accounts, caps how far
back to ingest via each account's ``latest_ingestion`` watermark, then runs the
full ingestion + preprocessing pipeline (including Stockfish enrichment).

No production data and no secrets are involved: games come from the public
Chess.com / Lichess APIs and land in the local Postgres + MinIO backends.

Usage (from repo root, with .env loaded):
    .venv/bin/python scripts/dev/hydrate_dev_data.py
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

from chess_teacher.pipelines.ingestion.main import run_ingestion_pipeline
from chess_teacher.pipelines.modes import PipelineMode
from chess_teacher.pipelines.preprocessing.main import run_preprocessing_pipeline
from chess_teacher.platform.account import Account, AccountPlatform
from chess_teacher.platform.user import User
from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.logging import get_logger

logger = get_logger()

# (username, platform, ingest-since) — cap volume to a representative window.
_ACCOUNTS: list[tuple[str, AccountPlatform, datetime]] = [
    ("RebeccaHarris", AccountPlatform.LICHESS, datetime(2025, 9, 11, tzinfo=UTC)),
    ("ikbendaniel", AccountPlatform.CHESS_COM, datetime(2026, 6, 1, tzinfo=UTC)),
]

_DEV_ST_USER = {
    "sub": "chess-teacher-dev-user",
    "provider": "google",
    "email": "dev@chess-teacher.local",
    "name": "Dev User",
    "email_verified": True,
}


def _ensure_dev_user(db_client) -> User:
    user = User.from_st_user(_DEV_ST_USER)
    user.save_new_to_db(db_client)
    logger.info("Dev user ready user_id=%s", user.user_id)
    return user


def _ensure_account(db_client, user: User, username: str, platform: AccountPlatform,
                    since: datetime) -> Account:
    account = Account.from_username_and_platform(username, platform)
    user.link_account(db_client, account)
    # Pre-set the ingestion watermark so the adapter only fetches games since `since`.
    account.upsert_field(db_client, "latest_ingestion", since)
    logger.info(
        "Linked account %s (since=%s) account_id=%s",
        account.format_label(), since.date(), account.account_id,
    )
    return account


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default=PipelineMode.INCREMENTAL,
                        type=PipelineMode, choices=list(PipelineMode))
    args = parser.parse_args()

    db_client = get_db_client()
    user = _ensure_dev_user(db_client)

    accounts = [
        _ensure_account(db_client, user, username, platform, since)
        for username, platform, since in _ACCOUNTS
    ]

    for account in accounts:
        logger.info("=== Ingesting %s ===", account.format_label())
        ing = run_ingestion_pipeline(user.user_id, account, mode=args.mode)
        logger.info("Ingestion result for %s: %s", account.format_label(), ing.result.value)

        logger.info("=== Preprocessing %s (Stockfish enrichment) ===", account.format_label())
        pre = run_preprocessing_pipeline(user.user_id, account, mode=args.mode)
        logger.info("Preprocessing result for %s: %s", account.format_label(), pre.result.value)

    logger.info("Hydration complete for user_id=%s", user.user_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
