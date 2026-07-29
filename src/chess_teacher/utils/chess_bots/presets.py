from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from chess_teacher.utils.chess_bots.base import ChessBot
from chess_teacher.utils.chess_bots.random_bot import RandomBot
from chess_teacher.utils.chess_bots.stockfish_bot import StockfishBot


@dataclass(frozen=True, slots=True)
class BotPreset:
    """Named opponent profile shown on the Play page."""

    key: str
    label: str
    description: str
    factory: Callable[[], ChessBot]


BOT_PRESETS: tuple[BotPreset, ...] = (
    BotPreset(
        key="random",
        label="Random",
        description="Random legal moves.",
        factory=RandomBot,
    ),
    BotPreset(
        key="stockfish_1",
        label="Beginner",
        description="Stockfish depth 1.",
        factory=lambda: StockfishBot(depth=1),
    ),
    BotPreset(
        key="stockfish_3",
        label="Intermediate",
        description="Stockfish depth 3.",
        factory=lambda: StockfishBot(depth=3),
    ),
    BotPreset(
        key="stockfish_10",
        label="Expert",
        description="Stockfish depth 10.",
        factory=lambda: StockfishBot(depth=10),
    ),
    BotPreset(
        key="stockfish_20",
        label="Stockfish",
        description="Stockfish depth 20.",
        factory=lambda: StockfishBot(depth=20),
    ),
)

BOT_PRESET_BY_KEY: dict[str, BotPreset] = {preset.key: preset for preset in BOT_PRESETS}


def get_bot_preset(key: str) -> BotPreset:
    try:
        return BOT_PRESET_BY_KEY[key]
    except KeyError as exc:
        raise KeyError(f"Unknown bot preset: {key!r}") from exc
