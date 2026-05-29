from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime

import polars as pl

from chess_teacher.ingestion.raw_games import RawGame
from chess_teacher.other.dataclasses import TimeControlCategory
from chess_teacher.platform.account import Account
from chess_teacher.utils.chess_utils import Color, Result
from chess_teacher.utils.db_client import DatabaseClient
from chess_teacher.utils.general_utils import quote_ident, quote_literal

_GAME_COLUMNS = [
    "game_id",
    "account_id",
    "color",
    "result",
    "variant",
    "time_control_initial",
    "time_control_increment",
    "start_time",
    "eco_code",
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


def load_games_for_accounts(
    db_client: DatabaseClient,
    account_ids: Sequence[str],
) -> pl.DataFrame:
    db_client.ensure_table(RawGame.get_metadata())
    if not account_ids:
        return pl.DataFrame()
    return db_client.read(
        RawGame.get_metadata(),
        columns=_GAME_COLUMNS,
        where=_where_account_ids(account_ids),
        order_by="start_time DESC NULLS LAST",
        as_polars=True,
    )


@dataclass(frozen=True)
class GameFilters:
    date_from: date | None = None
    date_to: date | None = None
    colors: frozenset[str] | None = None
    results: frozenset[str] | None = None
    variants: frozenset[str] | None = None
    account_ids: frozenset[str] | None = None
    time_controls: frozenset[str] | None = None


def build_rating_history(
    games: pl.DataFrame,
    accounts_by_id: dict[str, Account],
) -> pl.DataFrame:
    """Long-format rows for rating-over-time chart, one series per account and time control.

    ``series_id`` is the stable chart key (``account_id|time_control``).
    ``series_label`` is human-readable text for tooltips (username + time control).
    """
    usernames = {account_id: account.username for account_id, account in accounts_by_id.items()}
    account_labels = {
        account_id: account.format_label() for account_id, account in accounts_by_id.items()
    }
    return (
        with_time_control_class(games)
        .filter(pl.col("start_time").is_not_null() & pl.col("user_elo").is_not_null())
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
class ColorBreakdown:
    games: int
    wins: int
    win_rate_pct: float | None


@dataclass(frozen=True)
class GameStatisticsSummary:
    total_games: int
    wins: int
    draws: int
    losses: int
    no_result: int
    win_rate_pct: float | None
    avg_user_elo: float | None
    earliest_game: datetime | None
    latest_game: datetime | None
    result_counts: dict[str, int]
    games_by_account: dict[str, int]
    top_openings: list[tuple[str, int]]
    color_breakdown: dict[str, ColorBreakdown]


def _win_rate_pct(wins: int, total: int) -> float | None:
    if total == 0:
        return None
    return round(100.0 * wins / total, 1)


def _color_breakdown(
    df: pl.DataFrame,
    *,
    colors: tuple[str, ...] = (Color.WHITE.value, Color.BLACK.value),
) -> dict[str, ColorBreakdown]:
    breakdown: dict[str, ColorBreakdown] = {}
    for color in colors:
        color_df = df.filter(pl.col("color") == color)
        games = color_df.height
        wins = color_df.filter(pl.col("result") == Result.WIN.value).height
        breakdown[color] = ColorBreakdown(
            games=games,
            wins=wins,
            win_rate_pct=_win_rate_pct(wins, games),
        )
    return breakdown


def compute_summary(
    games: pl.DataFrame,
    accounts_by_id: dict[str, Account],
    *,
    color_breakdown_colors: tuple[str, ...] = (Color.WHITE.value, Color.BLACK.value),
) -> GameStatisticsSummary:
    total_games = games.height
    wins = games.filter(pl.col("result") == Result.WIN.value).height
    draws = games.filter(pl.col("result") == Result.DRAW.value).height
    losses = games.filter(pl.col("result") == Result.LOSS.value).height
    no_result = games.filter(pl.col("result") == Result.NO_RESULT.value).height
    decisive = wins + draws + losses

    avg_elo = games.select(pl.col("user_elo").drop_nulls().mean()).item()

    start_times = games.select(pl.col("start_time").drop_nulls())
    earliest = start_times.select(pl.col("start_time").min()).item()
    latest = start_times.select(pl.col("start_time").max()).item()

    result_counts = {
        label: games.filter(pl.col("result") == result).height
        for result, label in RESULT_LABELS.items()
    }

    account_labels = {
        account_id: account.format_label() for account_id, account in accounts_by_id.items()
    }
    account_frame = (
        games
        .select("account_id")
        .with_columns(
            pl
            .col("account_id")
            .replace_strict(account_labels, default="Unknown account")
            .alias("account")
        )
        .group_by("account")
        .len()
        .sort("len", descending=True)
    )
    games_by_account = {row["account"]: row["len"] for row in account_frame.iter_rows(named=True)}

    opening_frame = (
        games
        .filter(pl.col("eco_code").is_not_null())
        .group_by("eco_code")
        .len()
        .sort("len", descending=True)
        .head(5)
    )
    top_openings = [(row["eco_code"], row["len"]) for row in opening_frame.iter_rows(named=True)]

    return GameStatisticsSummary(
        total_games=total_games,
        wins=wins,
        draws=draws,
        losses=losses,
        no_result=no_result,
        win_rate_pct=_win_rate_pct(wins, decisive),
        avg_user_elo=round(avg_elo, 0) if avg_elo is not None else None,
        earliest_game=earliest,
        latest_game=latest,
        result_counts=result_counts,
        games_by_account=games_by_account,
        top_openings=top_openings,
        color_breakdown=_color_breakdown(games, colors=color_breakdown_colors),
    )
