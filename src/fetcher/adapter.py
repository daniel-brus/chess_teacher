from datetime import UTC, datetime

import requests

BASE_URL = "https://api.chess.com/pub"
HEADERS = {"User-Agent": "chess_teacher/1.0"}


def get_archives(username: str) -> list[str]:
    """Haal alle beschikbare maandelijkse archief-URLs op."""
    url = f"{BASE_URL}/player/{username}/games/archives"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json()["archives"]


def get_games_for_month(archive_url: str) -> list[dict]:
    """Haal alle potjes op voor een specifieke maand."""
    response = requests.get(archive_url, headers=HEADERS)
    response.raise_for_status()
    return response.json()["games"]


def get_recent_games(username: str, months: int = 1) -> list[dict]:
    """Haal potjes op van de laatste N maanden."""
    archives = get_archives(username)
    recent = archives[-months:]

    games = []
    for archive_url in recent:
        games.extend(get_games_for_month(archive_url))

    return games


def get_new_games(username: str, since: int | None = None) -> list[dict]:
    """
    Haal alleen nieuwe potjes op sinds unix timestamp `since`.
    Als since=None, haal de huidige maand op.
    """
    archives = get_archives(username)

    if since is None:
        # eerste run: alleen huidige maand
        to_fetch = archives[-1:]
    else:
        # filter archives op basis van since timestamp
        # archive urls eindigen op YYYY/MM — vergelijk met since
        to_fetch = _archives_since(archives, since)

    games = []
    for archive_url in to_fetch:
        month_games = get_games_for_month(archive_url)
        if since:
            month_games = [g for g in month_games if g["end_time"] > since]
        games.extend(month_games)

    return games


def _archives_since(archives: list[str], since: int) -> list[str]:
    """Filter archives zodat we alleen maanden ophalen die relevant zijn."""
    since_dt = datetime.fromtimestamp(since, tz=UTC)

    relevant = []
    for url in archives:
        # url eindigt op .../YYYY/MM
        parts = url.split("/")
        year, month = int(parts[-2]), int(parts[-1])
        archive_dt = datetime(year, month, 1, tzinfo=UTC)
        if archive_dt >= datetime(since_dt.year, since_dt.month, 1, tzinfo=UTC):
            relevant.append(url)

    return relevant
