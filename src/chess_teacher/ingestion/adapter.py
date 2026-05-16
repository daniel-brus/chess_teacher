import calendar
import json
from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta

import requests

from chess_teacher.platform.account import Account, AccountPlatform
from chess_teacher.utils.exception_utils import AdapterError
from chess_teacher.utils.general_utils import get_current_datetime
from chess_teacher.utils.logging_utils import get_logger


class Adapter(ABC):
    """Adapter for retrieving data from an API stream."""

    def __init__(self, account: Account):
        self.account = account
        self.logger = get_logger()

    @abstractmethod
    def _get_base_url(self, **kwargs) -> str:
        """Get the base URL for the API request."""
        pass

    @abstractmethod
    def _get_headers(self) -> dict:
        """Get the headers for the API request."""
        pass

    def _get_response(
        self, *, params: dict | None = None, stream: bool = False, **kwargs
    ) -> requests.Response:
        """Shared GET request with error handling and timeout."""
        try:
            self.logger.info(f"Getting response from {self._get_base_url(**kwargs)}.")
            response = requests.get(
                url=self._get_base_url(**kwargs),
                headers=self._get_headers(),
                params=params,
                timeout=30,
                stream=stream,
            )
            self.logger.info(f"Response status: {response.status_code}.")
            response.raise_for_status()
        except Exception as e:
            self.logger.log_and_raise(AdapterError(f"Error getting response: {e}"))
        return response

    @abstractmethod
    def get_records(self, since: datetime | None = None) -> list[dict]:
        """Get a list of records from the API stream."""
        pass


# ---------------------------------------------------------------------------
# API-specific adapters
# ---------------------------------------------------------------------------


class ChessComAdapter(Adapter):
    """
    Adapter for retrieving chess games from the Chess.com API.

    Endpoint: GET /pub/player/{username}/games/{year}/{month}
    Format:   JSON
    Auth:     None required for public games.

    Year and month are path segments, not query params.
    """

    _BASE_URL = "https://api.chess.com/pub"
    _LAUNCH_YEAR = 2005

    def __init__(self, account: Account) -> None:
        super().__init__(account)

    def _get_base_url(self, **kwargs) -> str:
        """
        Build URL for a specific year/month.

        Required kwargs: year (int), month (int)
        """
        year, month = _validate_year_month(kwargs, self._LAUNCH_YEAR)
        return f"{self._BASE_URL}/player/{self.account.username}/games/{year}/{month:02d}"

    def _get_headers(self) -> dict:
        return {
            "Accept": "application/json",
        }

    def get_records(self, since: datetime | None = None) -> list[dict]:
        """
        Fetch all games for the account since `since`.
        Iterates over all months from `since` to now, one request per month.
        Returns a flat list of game dicts (raw Chess.com JSON).
        """
        records = []
        for year, month in _get_months_since(since or datetime(self._LAUNCH_YEAR, 1, 1)):
            response = self._get_response(year=year, month=month)
            data = response.json()
            records.extend(data.get("games", []))
        return records


class LichessAdapter(Adapter):
    """
    Adapter for retrieving chess games from the Lichess API.

    Endpoint: GET /api/games/user/{username}
    Format:   NDJSON (newline-delimited JSON)
    Auth:     None required for public games.

    `since` and `until` are passed as Unix timestamps in milliseconds
    via query params — no path segments needed.
    """

    _BASE_URL = "https://lichess.org/api"

    def __init__(self, account: Account) -> None:
        super().__init__(account)

    def _get_base_url(self, **kwargs) -> str:
        return f"{self._BASE_URL}/games/user/{self.account.username}"

    def _get_headers(self) -> dict:
        return {
            "Accept": "application/x-ndjson",
        }

    def _parse_ndjson(self, response: requests.Response) -> list[dict]:
        """Parse a streaming NDJSON response into a list of dicts."""
        records = []
        try:
            self.logger.info(f"Parsing NDJSON response from {response.url}.")
            for line in response.iter_lines():
                if line:
                    records.append(json.loads(line))
        except Exception as e:
            self.logger.log_and_raise(AdapterError(f"Error parsing NDJSON response: {e}"))
        return records

    def get_records(self, since: datetime | None = None) -> list[dict]:
        """
        Fetch all games for the account since `since`.
        Returns a flat list of game dicts (raw Lichess JSON).
        """
        params = {
            "pgnInJson": "true",  # include PGN inside JSON instead of separate
            "opening": "true",  # include opening info
            "until": _to_unix_ms(get_current_datetime()),
        }

        if since is not None:
            params["since"] = _to_unix_ms(since)

        response = self._get_response(params=params, stream=True)
        result = self._parse_ndjson(response)
        self.logger.info(f"Parsed {len(result)} records from {response.url}.")
        return result


class AdapterFactory:
    @classmethod
    def from_account(cls, account: Account) -> Adapter:
        match account.platform:
            case AccountPlatform.CHESS_COM:
                return ChessComAdapter(account)
            case AccountPlatform.LICHESS:
                return LichessAdapter(account)
            case _:
                raise ValueError(f"No adapter for platform '{account.platform}'")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_year_month(kwargs: dict, launch_year: int = 0) -> tuple[int, int]:
    """
    Validate and extract year and month from kwargs.

    Raises:
        ValueError: if year or month is missing, not an int, or out of valid range.
    """
    now = get_current_datetime()

    if "year" not in kwargs:
        raise ValueError("Missing required kwarg 'year' for Chess.com URL.")
    if "month" not in kwargs:
        raise ValueError("Missing required kwarg 'month' for Chess.com URL.")

    year = kwargs["year"]
    month = kwargs["month"]

    if not isinstance(year, int):
        raise ValueError(f"'year' must be an int, got {type(year).__name__}.")
    if not isinstance(month, int):
        raise ValueError(f"'month' must be an int, got {type(month).__name__}.")

    if launch_year is not None and year < launch_year:
        raise ValueError(f"'year' must be >= {launch_year} (launch year), got {year}.")
    if year > now.year:
        raise ValueError(f"'year' must be <= current year ({now.year}), got {year}.")
    if not 1 <= month <= 12:
        raise ValueError(f"'month' must be between 1 and 12, got {month}.")
    if year == now.year and month > now.month:
        raise ValueError(
            f"Cannot request future month {year}/{month:02d} "
            f"(current: {now.year}/{now.month:02d})."
        )

    return year, month


def _get_months_since(since: datetime) -> list[tuple[int, int]]:
    now = get_current_datetime()
    months = []
    current = since.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while (current.year, current.month) <= (now.year, now.month):
        months.append((current.year, current.month))
        # Laatste dag van de huidige maand + 1 dag = eerste dag van volgende maand
        last_day = calendar.monthrange(current.year, current.month)[1]
        current = current.replace(day=last_day) + timedelta(days=1)
    return months


def _to_unix_ms(dt: datetime) -> int:
    """Convert a datetime to Unix timestamp in milliseconds (required by Lichess API)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)
