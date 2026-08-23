from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from chess_teacher.utils.chess_bots.base import ChessBot
from chess_teacher.utils.chess_bots.random_bot import RandomBot
from chess_teacher.utils.chess_bots.stockfish_bot import StockfishBot
from chess_teacher.utils.db.client import DatabaseClient
from chess_teacher.utils.exception_utils import ConfigError, DatabaseError, MetadataError
from chess_teacher.utils.logging import get_logger

BASELINE_PRESET_PREFIX = "baseline:"
STOCKFISH_PRESET_PREFIX = "stockfish:"
logger = get_logger()

STOCKFISH_DEPTH_MIN = 1
STOCKFISH_DEPTH_MAX = 20
STOCKFISH_DEPTH_DEFAULT = 3


class OpponentCategory(StrEnum):
    STOCKFISH = "stockfish"
    BASELINE = "baseline"
    PERSONAL = "personal"
    OTHER = "other"


OPPONENT_CATEGORY_LABELS: dict[OpponentCategory, str] = {
    OpponentCategory.STOCKFISH: "Stockfish",
    OpponentCategory.BASELINE: "Baseline bot",
    OpponentCategory.PERSONAL: "Personal bot",
    OpponentCategory.OTHER: "Other",
}


@dataclass(frozen=True, slots=True)
class BotPreset:
    """Named opponent profile shown on the Play page."""

    key: str
    label: str
    description: str
    factory: Callable[[], ChessBot]


def stockfish_preset_key(depth: int) -> str:
    return f"{STOCKFISH_PRESET_PREFIX}{int(depth)}"


def baseline_preset_key(version: str) -> str:
    return f"{BASELINE_PRESET_PREFIX}{version}"


def _stockfish_factory(depth: int) -> Callable[[], ChessBot]:
    def factory() -> ChessBot:
        return StockfishBot(depth=depth)

    return factory


def _stockfish_preset(depth: int) -> BotPreset:
    depth = int(depth)
    return BotPreset(
        key=stockfish_preset_key(depth),
        label=f"Stockfish (depth {depth})",
        description=f"Stockfish search depth {depth}.",
        factory=_stockfish_factory(depth),
    )


def _baseline_factory(model_uri: str, version: str) -> Callable[[], ChessBot]:
    def factory() -> ChessBot:
        from chess_teacher.utils.chess_bots.neural_baseline_bot import NeuralBaselineBot

        return NeuralBaselineBot(model_uri=model_uri, version=version)

    return factory


# Legacy fixed Stockfish keys (in-progress games / bookmarks). Prefer stockfish:N.
_LEGACY_STOCKFISH: dict[str, int] = {
    "stockfish_1": 1,
    "stockfish_3": 3,
    "stockfish_10": 10,
    "stockfish_20": 20,
}

BOT_PRESETS: tuple[BotPreset, ...] = (
    BotPreset(
        key="random",
        label="Random",
        description="Random legal moves.",
        factory=RandomBot,
    ),
    *(_stockfish_preset(d) for d in (1, 3, 10, 20)),
    # Keep old keys resolvable for existing session state.
    *(
        BotPreset(
            key=key,
            label=f"Stockfish (depth {depth})",
            description=f"Stockfish search depth {depth}.",
            factory=_stockfish_factory(depth),
        )
        for key, depth in _LEGACY_STOCKFISH.items()
    ),
)

BOT_PRESET_BY_KEY: dict[str, BotPreset] = {preset.key: preset for preset in BOT_PRESETS}


def list_baseline_presets(db_client: DatabaseClient) -> list[BotPreset]:
    """Playable baseline policies: production + archived (once-promoted) only."""
    from chess_teacher.pipelines.neural_network.models import (
        BaselineModel,
        BaselineModelStatus,
    )

    playable_statuses = {BaselineModelStatus.PRODUCTION, BaselineModelStatus.ARCHIVED}
    rows = [
        row
        for row in BaselineModel.fetch_all_ordered(db_client)
        if row.status in playable_statuses and row.looks_like_candidate_style() and row.model_uri
    ]
    # Current production first, then newer archived.
    rows.sort(
        key=lambda r: (
            0 if r.status == BaselineModelStatus.PRODUCTION else 1,
            -(r.trained_at.timestamp() if r.trained_at is not None else 0.0),
        )
    )

    presets: list[BotPreset] = []
    for row in rows:
        model_uri = row.model_uri
        if model_uri is None:
            continue
        status_label = "current" if row.status == BaselineModelStatus.PRODUCTION else "archived"
        presets.append(
            BotPreset(
                key=baseline_preset_key(row.version),
                label=f"Baseline {row.version}",
                description=f"Neural candidate-style ({status_label}).",
                factory=_baseline_factory(model_uri, row.version),
            )
        )
    return presets


def list_other_presets() -> list[BotPreset]:
    """Non-engine / non-neural toys (Random, …)."""
    return [BOT_PRESET_BY_KEY["random"]]


def list_play_presets(db_client: DatabaseClient | None = None) -> list[BotPreset]:
    """Flat list for caption / resolve helpers (static + playable baselines)."""
    presets: list[BotPreset] = [
        BOT_PRESET_BY_KEY["random"],
        *(_stockfish_preset(d) for d in (1, 3, 10, 20)),
    ]
    if db_client is None:
        return presets
    try:
        presets.extend(list_baseline_presets(db_client))
    except (DatabaseError, MetadataError, ConfigError):
        logger.exception("Failed to load baseline bot presets; using static bots only")
        return [
            BOT_PRESET_BY_KEY["random"],
            *(_stockfish_preset(d) for d in (1, 3, 10, 20)),
        ]
    return presets


def get_bot_preset(key: str, *, db_client: DatabaseClient | None = None) -> BotPreset:
    if key in BOT_PRESET_BY_KEY:
        return BOT_PRESET_BY_KEY[key]

    if key.startswith(STOCKFISH_PRESET_PREFIX):
        raw = key.removeprefix(STOCKFISH_PRESET_PREFIX)
        try:
            depth = int(raw)
        except ValueError as exc:
            raise KeyError(f"Unknown stockfish bot preset: {key!r}") from exc
        if not STOCKFISH_DEPTH_MIN <= depth <= STOCKFISH_DEPTH_MAX:
            raise KeyError(
                f"Stockfish depth out of range ({STOCKFISH_DEPTH_MIN}-{STOCKFISH_DEPTH_MAX}): "
                f"{key!r}"
            )
        return _stockfish_preset(depth)

    if key.startswith(BASELINE_PRESET_PREFIX):
        version = key.removeprefix(BASELINE_PRESET_PREFIX)
        client = db_client
        if client is None:
            from chess_teacher.utils.db.client import get_db_client

            client = get_db_client()
        for preset in list_baseline_presets(client):
            if preset.key == key:
                return preset
        raise KeyError(f"Unknown baseline bot preset: {key!r} (version={version!r})")

    raise KeyError(f"Unknown bot preset: {key!r}")


def category_for_preset_key(key: str) -> OpponentCategory:
    """Best-effort category for restoring setup UI from a preset key."""
    if key.startswith(STOCKFISH_PRESET_PREFIX) or key in _LEGACY_STOCKFISH:
        return OpponentCategory.STOCKFISH
    if key.startswith(BASELINE_PRESET_PREFIX):
        return OpponentCategory.BASELINE
    if key == "random" or key in {p.key for p in list_other_presets()}:
        return OpponentCategory.OTHER
    return OpponentCategory.OTHER


def depth_from_preset_key(key: str) -> int | None:
    if key.startswith(STOCKFISH_PRESET_PREFIX):
        try:
            return int(key.removeprefix(STOCKFISH_PRESET_PREFIX))
        except ValueError:
            return None
    return _LEGACY_STOCKFISH.get(key)
