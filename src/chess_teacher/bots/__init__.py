"""Chess opponent bots for interactive play."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chess_teacher.bots.base import ChessBot
from chess_teacher.bots.presets import (
    BASELINE_PRESET_PREFIX,
    BASELINE_TEMPERATURE_DEFAULT,
    BASELINE_TEMPERATURE_MAX,
    BASELINE_TEMPERATURE_MIN,
    BASELINE_TEMPERATURE_STEP,
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
    invalidate_baseline_presets_cache,
    list_baseline_presets,
    list_other_presets,
    list_play_presets,
    stockfish_preset_key,
)
from chess_teacher.bots.random_bot import RandomBot
from chess_teacher.bots.stockfish_bot import StockfishBot

if TYPE_CHECKING:
    from chess_teacher.bots.move_analysis import BotMoveAnalysis, BotMoveCandidateRow
    from chess_teacher.bots.neural_baseline_bot import NeuralBaselineBot

__all__ = [
    "BASELINE_PRESET_PREFIX",
    "BASELINE_TEMPERATURE_DEFAULT",
    "BASELINE_TEMPERATURE_MAX",
    "BASELINE_TEMPERATURE_MIN",
    "BASELINE_TEMPERATURE_STEP",
    "BOT_PRESETS",
    "BOT_PRESET_BY_KEY",
    "OPPONENT_CATEGORY_LABELS",
    "STOCKFISH_DEPTH_DEFAULT",
    "STOCKFISH_DEPTH_MAX",
    "STOCKFISH_DEPTH_MIN",
    "STOCKFISH_PRESET_PREFIX",
    "BotMoveAnalysis",
    "BotMoveCandidateRow",
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
    "invalidate_baseline_presets_cache",
    "list_baseline_presets",
    "list_other_presets",
    "list_play_presets",
    "stockfish_preset_key",
]


def __getattr__(name: str) -> Any:
    if name == "NeuralBaselineBot":
        from chess_teacher.bots.neural_baseline_bot import NeuralBaselineBot

        return NeuralBaselineBot
    if name == "BotMoveAnalysis":
        from chess_teacher.bots.move_analysis import BotMoveAnalysis

        return BotMoveAnalysis
    if name == "BotMoveCandidateRow":
        from chess_teacher.bots.move_analysis import BotMoveCandidateRow

        return BotMoveCandidateRow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
