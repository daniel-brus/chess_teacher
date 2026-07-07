from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime

import polars as pl

from chess_teacher.other.dataclasses import TimeControlCategory
from chess_teacher.pipelines.preprocessing.games import Game
from chess_teacher.platform.account import Account
from chess_teacher.utils.cache_utils import get_cache_client
from chess_teacher.utils.chess_utils import Color, Result
from chess_teacher.utils.db.client import DatabaseClient
from chess_teacher.utils.general_utils import quote_ident, quote_literal
from chess_teacher.utils.logging import get_logger

logger = get_logger()

_GAME_COLUMNS = [
    "game_id",
    "account_id",
    "color",
    "result",
    "variant",
    "time_control_initial",
    "time_control_increment",
    "start_time",
    "opening_family",
    "user_elo",
    "opponent_elo",
]

RESULT_LABELS = {
    Result.WIN.value: "Win",
    Result.DRAW.value: "Draw",
    Result.LOSS.value: "Loss",
    Result.NO_RESULT.value: "No result",
}

TIME_CONTROL_CLASSES = tuple(category.value for category in TimeControlCategory)
_TIME_CONTROL_SORT_ORDER = {
    category.value: index for index, category in enumerate(TimeControlCategory)
}


def with_time_control_class(games: pl.DataFrame) -> pl.DataFrame:
    if "time_control" in games.columns:
        return games
    return games.with_columns(
        pl
        .struct("time_control_initial", "time_control_increment")
        .map_elements(
            lambda row: (
                TimeControlCategory.from_initial_and_increment(
                    row["time_control_initial"],
                    row["time_control_increment"],
                ).value
            ),
            return_dtype=pl.Utf8,
        )
        .alias("time_control")
    )


def sorted_time_controls(values: Sequence[str]) -> list[str]:
    return sorted(
        values, key=lambda value: _TIME_CONTROL_SORT_ORDER.get(value, len(TIME_CONTROL_CLASSES))
    )


def _where_account_ids(account_ids: Sequence[str]) -> str:
    if not account_ids:
        return "FALSE"
    in_list = ", ".join(quote_literal(account_id) for account_id in account_ids)
    return f"{quote_ident('account_id')} IN ({in_list})"


def _decode_html_text_columns(df: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
    present = [column for column in columns if column in df.columns]
    if df.is_empty() or not present:
        return df
    return df.with_columns(
        pl
        .col(column)
        .map_elements(
            lambda value: html.unescape(value) if isinstance(value, str) else value,
            return_dtype=pl.Utf8,
        )
        .alias(column)
        for column in present
    )


def load_games_for_accounts(
    db_client: DatabaseClient,
    account_ids: Sequence[str],
    *,
    user_id: str | None = None,
) -> pl.DataFrame:
    if not account_ids:
        return pl.DataFrame()

    cache = get_cache_client()
    if cache is not None and user_id is not None:
        cached_games = cache.get_user_games(user_id)
        if cached_games is not None:
            return cached_games

    db_client.ensure_table(Game.get_metadata())
    logger.info(
        "Loading games from database user_id=%s account_count=%s",
        user_id,
        len(account_ids),
    )
    games = db_client.read(
        Game.get_metadata(),
        columns=_GAME_COLUMNS,
        where=_where_account_ids(account_ids),
        order_by="start_time DESC NULLS LAST",
        as_polars=True,
    )
    games = _decode_html_text_columns(games, ("opening_family",))
    logger.info(
        "Loaded games from database user_id=%s rows=%s",
        user_id,
        games.height,
    )

    if cache is not None and user_id is not None:
        cache.set_user_games(user_id, games)

    return games


@dataclass(frozen=True)
class GameFilters:
    date_from: date | None = None
    date_to: date | None = None
    colors: frozenset[str] | None = None
    results: frozenset[str] | None = None
    variants: frozenset[str] | None = None
    account_ids: frozenset[str] | None = None
    time_controls: frozenset[str] | None = None


def _with_account_series_columns(
    games: pl.DataFrame,
    accounts_by_id: dict[str, Account],
) -> pl.DataFrame:
    """Add ``series_id`` / ``series_label`` (account + time control), same as rating chart."""
    usernames = {account_id: account.username for account_id, account in accounts_by_id.items()}
    account_labels = {
        account_id: account.format_label() for account_id, account in accounts_by_id.items()
    }
    return (
        with_time_control_class(games)
        .with_columns(
            pl.col("account_id").replace_strict(usernames, default="Unknown").alias("username"),
            pl
            .col("account_id")
            .replace_strict(account_labels, default="Unknown account")
            .alias("account"),
        )
        .with_columns(
            (pl.col("account_id") + pl.lit("|") + pl.col("time_control")).alias("series_id"),
            (pl.col("username") + " (" + pl.col("time_control") + ")").alias("series_label"),
        )
    )


def build_rating_history(
    games: pl.DataFrame,
    accounts_by_id: dict[str, Account],
) -> pl.DataFrame:
    """Long-format rows for rating-over-time chart, one series per account and time control.

    ``series_id`` is the stable chart key (``account_id|time_control``).
    ``series_label`` is human-readable text for tooltips (username + time control).
    """
    return (
        _with_account_series_columns(games, accounts_by_id)
        .filter(pl.col("start_time").is_not_null() & pl.col("user_elo").is_not_null())
        .select(
            "start_time",
            "user_elo",
            "account_id",
            "username",
            "account",
            "time_control",
            "series_id",
            "series_label",
        )
        .sort("start_time")
    )


def get_dated_bounds(games: pl.DataFrame) -> tuple[date, date] | None:
    dated = games.filter(pl.col("start_time").is_not_null())
    if dated.is_empty():
        return None
    earliest = dated.select(pl.col("start_time").min()).item()
    latest = dated.select(pl.col("start_time").max()).item()
    return earliest.date(), latest.date()


def apply_filters(games: pl.DataFrame, filters: GameFilters) -> pl.DataFrame:
    filtered = with_time_control_class(games)

    if filters.time_controls is not None:
        filtered = filtered.filter(pl.col("time_control").is_in(list(filters.time_controls)))

    if filters.colors is not None:
        filtered = filtered.filter(pl.col("color").is_in(list(filters.colors)))

    if filters.results is not None:
        filtered = filtered.filter(pl.col("result").is_in(list(filters.results)))

    if filters.variants is not None:
        filtered = filtered.filter(pl.col("variant").is_in(list(filters.variants)))

    if filters.account_ids is not None:
        filtered = filtered.filter(pl.col("account_id").is_in(list(filters.account_ids)))

    if filters.date_from is not None:
        filtered = filtered.filter(
            pl.col("start_time").is_not_null()
            & (pl.col("start_time").dt.date() >= filters.date_from)
        )
    if filters.date_to is not None:
        filtered = filtered.filter(
            pl.col("start_time").is_not_null() & (pl.col("start_time").dt.date() <= filters.date_to)
        )

    return filtered


@dataclass(frozen=True)
class PeakRating:
    rating: int
    account_label: str
    time_control: str
    game_date: datetime | None


@dataclass(frozen=True)
class HighestOpponentBeat:
    opponent_elo: int
    game_date: datetime | None


@dataclass(frozen=True)
class FavoriteTimeControl:
    time_control: str
    share_pct: float


@dataclass(frozen=True)
class LongestWinStreak:
    length: int
    start_date: datetime | None
    end_date: datetime | None


@dataclass(frozen=True)
class ColorBreakdown:
    games: int
    rated_games: int
    wins: int
    draws: int
    losses: int
    win_rate_pct: float | None
    loss_rate_pct: float | None
    favorite_opening: str | None
    favorite_opening_share_pct: float | None
    longest_win_streak: LongestWinStreak
    best_opponent_beat: HighestOpponentBeat | None


@dataclass(frozen=True)
class AccountCategoryGameCount:
    series_id: str
    series_label: str
    account_id: str
    games: int


@dataclass(frozen=True)
class GameStatisticsSummary:
    total_games: int
    rated_games: int
    wins: int
    draws: int
    losses: int
    no_result: int
    win_rate_pct: float | None
    peak_rating: PeakRating | None
    highest_opponent_beat: HighestOpponentBeat | None
    first_game: datetime | None
    favorite_time_control: FavoriteTimeControl | None
    longest_win_streak: LongestWinStreak
    result_counts: dict[str, int]
    games_by_account_and_category: tuple[AccountCategoryGameCount, ...]
    top_openings: list[tuple[str, int]]
    color_breakdown: dict[str, ColorBreakdown]


def _win_rate_pct(wins: int, total: int) -> float | None:
    if total == 0:
        return None
    return round(100.0 * wins / total, 1)


def _loss_rate_pct(losses: int, decisive: int) -> float | None:
    if decisive == 0:
        return None
    return round(100.0 * losses / decisive, 1)


def _compute_favorite_opening(games: pl.DataFrame) -> tuple[str | None, float | None]:
    if games.is_empty():
        return None, None
    with_family = games.filter(pl.col("opening_family").is_not_null())
    if with_family.is_empty():
        return None, None
    top_row = (
        with_family
        .group_by("opening_family")
        .len()
        .sort("len", descending=True)
        .head(1)
        .iter_rows(named=True)
        .__next__()
    )
    share_pct = round(100.0 * top_row["len"] / games.height, 1)
    return top_row["opening_family"], share_pct


def _account_labels(accounts_by_id: dict[str, Account]) -> dict[str, str]:
    return {account_id: account.format_label() for account_id, account in accounts_by_id.items()}


def _compute_peak_rating(
    games: pl.DataFrame,
    accounts_by_id: dict[str, Account],
) -> PeakRating | None:
    rated = with_time_control_class(games).filter(pl.col("user_elo").is_not_null())
    if rated.is_empty():
        return None
    row = (
        rated
        .sort(["user_elo", "start_time"], descending=[True, True])
        .head(1)
        .iter_rows(named=True)
        .__next__()
    )
    account_label = _account_labels(accounts_by_id).get(row["account_id"], "Unknown account")
    return PeakRating(
        rating=int(row["user_elo"]),
        account_label=account_label,
        time_control=row["time_control"],
        game_date=row["start_time"],
    )


def _compute_highest_opponent_beat(games: pl.DataFrame) -> HighestOpponentBeat | None:
    wins = games.filter(
        (pl.col("result") == Result.WIN.value) & pl.col("opponent_elo").is_not_null()
    )
    if wins.is_empty():
        return None
    row = (
        wins
        .sort(["opponent_elo", "start_time"], descending=[True, True])
        .head(1)
        .iter_rows(named=True)
        .__next__()
    )
    return HighestOpponentBeat(
        opponent_elo=int(row["opponent_elo"]),
        game_date=row["start_time"],
    )


def _compute_favorite_time_control(games: pl.DataFrame) -> FavoriteTimeControl | None:
    if games.is_empty():
        return None
    top_row = (
        with_time_control_class(games)
        .group_by("time_control")
        .len()
        .sort("len", descending=True)
        .head(1)
        .iter_rows(named=True)
        .__next__()
    )
    share_pct = round(100.0 * top_row["len"] / games.height, 1)
    return FavoriteTimeControl(time_control=top_row["time_control"], share_pct=share_pct)


def _compute_longest_win_streak(games: pl.DataFrame) -> LongestWinStreak:
    dated = games.filter(pl.col("start_time").is_not_null()).sort("start_time")
    if dated.is_empty():
        return LongestWinStreak(length=0, start_date=None, end_date=None)

    best_length = 0
    best_start: datetime | None = None
    best_end: datetime | None = None
    current_length = 0
    current_start: datetime | None = None

    for row in dated.iter_rows(named=True):
        if row["result"] == Result.WIN.value:
            if current_length == 0:
                current_start = row["start_time"]
            current_length += 1
            if current_length > best_length:
                best_length = current_length
                best_start = current_start
                best_end = row["start_time"]
        else:
            current_length = 0
            current_start = None

    return LongestWinStreak(length=best_length, start_date=best_start, end_date=best_end)


def _color_breakdown(
    df: pl.DataFrame,
    *,
    colors: tuple[str, ...] = (Color.WHITE.value, Color.BLACK.value),
) -> dict[str, ColorBreakdown]:
    breakdown: dict[str, ColorBreakdown] = {}
    for color in colors:
        color_df = df.filter(pl.col("color") == color)
        games = color_df.height
        rated_games = color_df.filter(pl.col("user_elo").is_not_null()).height
        wins = color_df.filter(pl.col("result") == Result.WIN.value).height
        draws = color_df.filter(pl.col("result") == Result.DRAW.value).height
        losses = color_df.filter(pl.col("result") == Result.LOSS.value).height
        decisive = wins + draws + losses
        favorite_opening, favorite_opening_share_pct = _compute_favorite_opening(color_df)
        breakdown[color] = ColorBreakdown(
            games=games,
            rated_games=rated_games,
            wins=wins,
            draws=draws,
            losses=losses,
            win_rate_pct=_win_rate_pct(wins, decisive),
            loss_rate_pct=_loss_rate_pct(losses, decisive),
            favorite_opening=favorite_opening,
            favorite_opening_share_pct=favorite_opening_share_pct,
            longest_win_streak=_compute_longest_win_streak(color_df),
            best_opponent_beat=_compute_highest_opponent_beat(color_df),
        )
    return breakdown


def compute_summary(
    games: pl.DataFrame,
    accounts_by_id: dict[str, Account],
    *,
    color_breakdown_colors: tuple[str, ...] = (Color.WHITE.value, Color.BLACK.value),
) -> GameStatisticsSummary:
    total_games = games.height
    rated_games = games.filter(pl.col("user_elo").is_not_null()).height
    wins = games.filter(pl.col("result") == Result.WIN.value).height
    draws = games.filter(pl.col("result") == Result.DRAW.value).height
    losses = games.filter(pl.col("result") == Result.LOSS.value).height
    no_result = games.filter(pl.col("result") == Result.NO_RESULT.value).height
    decisive = wins + draws + losses

    start_times = games.select(pl.col("start_time").drop_nulls())
    first_game = start_times.select(pl.col("start_time").min()).item()

    result_counts = {
        label: games.filter(pl.col("result") == result).height
        for result, label in RESULT_LABELS.items()
    }

    series_frame = (
        _with_account_series_columns(games, accounts_by_id)
        .group_by("series_id", "series_label", "account_id")
        .len()
        .sort("len", descending=True)
    )
    games_by_account_and_category = tuple(
        AccountCategoryGameCount(
            series_id=row["series_id"],
            series_label=row["series_label"],
            account_id=row["account_id"],
            games=row["len"],
        )
        for row in series_frame.iter_rows(named=True)
    )

    opening_frame = (
        games
        .filter(pl.col("opening_family").is_not_null())
        .group_by("opening_family")
        .len()
        .sort("len", descending=True)
        .head(5)
    )
    top_openings = [
        (row["opening_family"], row["len"]) for row in opening_frame.iter_rows(named=True)
    ]

    return GameStatisticsSummary(
        total_games=total_games,
        rated_games=rated_games,
        wins=wins,
        draws=draws,
        losses=losses,
        no_result=no_result,
        win_rate_pct=_win_rate_pct(wins, decisive),
        peak_rating=_compute_peak_rating(games, accounts_by_id),
        highest_opponent_beat=_compute_highest_opponent_beat(games),
        first_game=first_game,
        favorite_time_control=_compute_favorite_time_control(games),
        longest_win_streak=_compute_longest_win_streak(games),
        result_counts=result_counts,
        games_by_account_and_category=games_by_account_and_category,
        top_openings=top_openings,
        color_breakdown=_color_breakdown(games, colors=color_breakdown_colors),
    )
