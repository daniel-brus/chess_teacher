"""Chess.com opening slug lookup backed by ``other.raw_chess_com_openings``."""

from __future__ import annotations

import html
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import requests

from chess_teacher.other.dataclasses import RawChessComOpening
from chess_teacher.utils.logging import EnhancedLogger, get_logger

if TYPE_CHECKING:
    from chess_teacher.utils.db.client import DatabaseClient

logger = get_logger()

_CHESS_COM_OPENING_TITLE_RE = re.compile(
    r'property="og:title"\s+content="([^"]+)"',
    re.IGNORECASE,
)
_CHESS_COM_TITLE_SUFFIX_RE = re.compile(r"\s*-\s*Chess Openings.*$", re.IGNORECASE)
_CHESS_COM_REQUEST_HEADERS = {
    "User-Agent": "chess-teacher/1.0 (opening-slug-lookup)",
}
_DEFAULT_FETCH_DELAY_S = 0.25


@dataclass(frozen=True)
class SlugLookupRefreshResult:
    distinct_slugs: int
    already_cached: int
    fetched: int
    unresolved: int


def fetch_chess_com_opening_title(
    slug: str,
    *,
    session: requests.Session | None = None,
) -> str | None:
    """Return the human-readable opening title for a Chess.com openings slug."""
    http = session or requests
    try:
        response = http.get(
            f"https://www.chess.com/openings/{slug}",
            timeout=30,
            headers=_CHESS_COM_REQUEST_HEADERS,
        )
        response.raise_for_status()
    except requests.RequestException:
        return None

    match = _CHESS_COM_OPENING_TITLE_RE.search(response.text)
    if not match:
        return None

    title = html.unescape(_CHESS_COM_TITLE_SUFFIX_RE.sub("", match.group(1)).strip())
    if not title or title.lower().startswith("chess openings"):
        return None
    return title


def load_slug_title_lookup(db_client: DatabaseClient | None = None) -> dict[str, str]:
    """Load slug → opening title map from ``other.raw_chess_com_openings``."""
    from chess_teacher.utils.db.client import get_db_client

    client = db_client or get_db_client()
    metadata = RawChessComOpening.get_metadata()
    client.ensure_metadata(metadata)
    rows = client.read(metadata, columns=["slug", "name"])
    return {row["slug"]: html.unescape(row["name"]) for row in rows}


def collect_distinct_slugs_from_database(db_client: DatabaseClient) -> set[str]:
    """Collect every Chess.com opening slug seen in ``raw_games``."""
    from chess_teacher.pipelines.ingestion.raw_games import RawGame
    from chess_teacher.pipelines.ingestion.transformations import (
        ApplyChessComOpeningLookupTransformation,
    )

    metadata = RawGame.get_metadata()
    db_client.ensure_metadata(metadata)

    slugs: set[str] = set()
    slug_rows = db_client.read(
        metadata,
        columns=["chess_com_opening_slug"],
        where="chess_com_opening_slug IS NOT NULL",
        as_polars=True,
    )
    if not slug_rows.is_empty():
        slugs.update(slug_rows["chess_com_opening_slug"].drop_nulls().unique().to_list())

    pgn_rows = db_client.read(
        metadata,
        columns=["raw_pgn"],
        where=(
            "chess_com_opening_slug IS NULL AND raw_pgn IS NOT NULL AND raw_pgn ILIKE '%[ECOUrl%'"
        ),
        as_polars=True,
    )
    if not pgn_rows.is_empty():
        for raw_pgn in pgn_rows["raw_pgn"].drop_nulls().to_list():
            slug = ApplyChessComOpeningLookupTransformation._chess_com_opening_slug_from_pgn(
                raw_pgn
            )
            if slug:
                slugs.add(slug)

    return slugs


def refresh_missing_slug_titles(
    db_client: DatabaseClient,
    slugs: set[str],
    *,
    request_delay_s: float = _DEFAULT_FETCH_DELAY_S,
    logger: EnhancedLogger | None = None,
) -> SlugLookupRefreshResult:
    """
    Fetch Chess.com titles for slugs missing from ``other.raw_chess_com_openings``.

    Existing rows are left unchanged. Newly resolved titles are upserted.
    """
    log = logger or get_logger()
    lookup = load_slug_title_lookup(db_client)
    missing_slugs = sorted(slug for slug in slugs if not lookup.get(slug))
    already_cached = len(slugs) - len(missing_slugs)

    if not missing_slugs:
        log.info(
            "Chess.com opening lookup up to date (%s distinct slug(s), all cached).",
            len(slugs),
        )
        return SlugLookupRefreshResult(
            distinct_slugs=len(slugs),
            already_cached=already_cached,
            fetched=0,
            unresolved=0,
        )

    log.info(
        "Fetching %s Chess.com opening slug(s) (%s already cached, %s in lookup table)...",
        len(missing_slugs),
        already_cached,
        len(lookup),
    )

    metadata = RawChessComOpening.get_metadata()
    db_client.ensure_metadata(metadata)
    fetched = 0
    new_records: list[dict[str, str]] = []
    with requests.Session() as session:
        for slug in missing_slugs:
            title = fetch_chess_com_opening_title(slug, session=session)
            if title:
                new_records.append(
                    RawChessComOpening.from_slug_and_name(slug, title)._to_db_record()
                )
                fetched += 1
            time.sleep(request_delay_s)

    if new_records:
        db_client.merge(new_records, metadata)

    unresolved = len(missing_slugs) - fetched
    log.info(
        "Chess.com opening lookup refresh: %s fetched, %s unresolved, %s total cached.",
        fetched,
        unresolved,
        len(lookup) + fetched,
    )
    return SlugLookupRefreshResult(
        distinct_slugs=len(slugs),
        already_cached=already_cached,
        fetched=fetched,
        unresolved=unresolved,
    )


def refresh_slug_title_lookup_from_database(
    db_client: DatabaseClient,
    *,
    request_delay_s: float = _DEFAULT_FETCH_DELAY_S,
    logger: EnhancedLogger | None = None,
) -> SlugLookupRefreshResult:
    """Scan ``raw_games`` for slugs and refresh missing entries in the lookup table."""
    slugs = collect_distinct_slugs_from_database(db_client)
    if not slugs:
        log = logger or get_logger()
        log.info("No Chess.com opening slugs found in raw_games.")
        return SlugLookupRefreshResult(
            distinct_slugs=0,
            already_cached=0,
            fetched=0,
            unresolved=0,
        )
    return refresh_missing_slug_titles(
        db_client,
        slugs,
        request_delay_s=request_delay_s,
        logger=logger,
    )
