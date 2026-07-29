"""Chess opponent bots for interactive play."""

from chess_teacher.utils.chess_bots.base import ChessBot
from chess_teacher.utils.chess_bots.presets import (
    BOT_PRESET_BY_KEY,
    BOT_PRESETS,
    BotPreset,
    get_bot_preset,
)
from chess_teacher.utils.chess_bots.random_bot import RandomBot
from chess_teacher.utils.chess_bots.stockfish_bot import StockfishBot

__all__ = [
    "BOT_PRESETS",
    "BOT_PRESET_BY_KEY",
    "BotPreset",
    "ChessBot",
    "RandomBot",
    "StockfishBot",
    "get_bot_preset",
]
