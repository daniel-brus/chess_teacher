"""Chess opponent bots for interactive play."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chess_teacher.utils.chess_bots.base import ChessBot
from chess_teacher.utils.chess_bots.presets import (
    BASELINE_PRESET_PREFIX,
    BOT_PRESET_BY_KEY,
    BOT_PRESETS,
    OPPONENT_CATEGORY_LABELS,
    STOCKFISH_DEPTH_DEFAULT,
    STOCKFISH_DEPTH_MAX,
    STOCKFISH_DEPTH_MIN,
    STOCKFISH_PRESET_PREFIX,
    BotPreset,
    OpponentCategory,
    baseline_preset_key,
    category_for_preset_key,
    depth_from_preset_key,
    get_bot_preset,
    list_baseline_presets,
    list_other_presets,
    list_play_presets,
    stockfish_preset_key,
)
from chess_teacher.utils.chess_bots.random_bot import RandomBot
from chess_teacher.utils.chess_bots.stockfish_bot import StockfishBot

if TYPE_CHECKING:
    from chess_teacher.utils.chess_bots.neural_baseline_bot import NeuralBaselineBot

__all__ = [
    "BASELINE_PRESET_PREFIX",
    "BOT_PRESETS",
    "BOT_PRESET_BY_KEY",
    "OPPONENT_CATEGORY_LABELS",
    "STOCKFISH_DEPTH_DEFAULT",
    "STOCKFISH_DEPTH_MAX",
    "STOCKFISH_DEPTH_MIN",
    "STOCKFISH_PRESET_PREFIX",
    "BotPreset",
    "ChessBot",
    "NeuralBaselineBot",
    "OpponentCategory",
    "RandomBot",
    "StockfishBot",
    "baseline_preset_key",
    "category_for_preset_key",
    "depth_from_preset_key",
    "get_bot_preset",
    "list_baseline_presets",
    "list_other_presets",
    "list_play_presets",
    "stockfish_preset_key",
]


def __getattr__(name: str) -> Any:
    if name == "NeuralBaselineBot":
        from chess_teacher.utils.chess_bots.neural_baseline_bot import NeuralBaselineBot

        return NeuralBaselineBot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
