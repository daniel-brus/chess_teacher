from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from chess_teacher.bots.base import ChessBot
from chess_teacher.bots.random_bot import RandomBot
from chess_teacher.bots.stockfish_bot import StockfishBot
from chess_teacher.utils.db.client import DatabaseClient
from chess_teacher.utils.exception_utils import ConfigError, DatabaseError, MetadataError
from chess_teacher.utils.logging import get_logger

if TYPE_CHECKING:
    from chess_teacher.pipelines.neural_network.models import PersonalModel

BASELINE_PRESET_PREFIX = "baseline:"
PERSONAL_PRESET_PREFIX = "personal:"
STOCKFISH_PRESET_PREFIX = "stockfish:"
logger = get_logger()

# Play page calls preset lookup on every Streamlit rerun; cache DB-backed baselines briefly.
_BASELINE_PRESETS_CACHE_TTL_SEC = 60.0
_baseline_presets_cache: dict[int, tuple[list[BotPreset], float]] = {}
_personal_presets_cache: dict[tuple[int, str], tuple[list[BotPreset], bool, float]] = {}

STOCKFISH_DEPTH_MIN = 1
STOCKFISH_DEPTH_MAX = 20
STOCKFISH_DEPTH_DEFAULT = 3

BASELINE_TEMPERATURE_MIN = 0.0
BASELINE_TEMPERATURE_MAX = 2.0
BASELINE_TEMPERATURE_DEFAULT = 0.0
BASELINE_TEMPERATURE_STEP = 0.05


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


def personal_preset_key(user_id: str, version: str) -> str:
    return f"{PERSONAL_PRESET_PREFIX}{user_id}:{version}"


def parse_personal_preset_key(key: str) -> tuple[str, str]:
    """Return ``(user_id, version)`` from a ``personal:{user_id}:{version}`` key."""
    if not key.startswith(PERSONAL_PRESET_PREFIX):
        raise KeyError(f"Unknown personal bot preset: {key!r}")
    user_id, sep, version = key.removeprefix(PERSONAL_PRESET_PREFIX).rpartition(":")
    if not sep or not user_id or not version:
        raise KeyError(f"Unknown personal bot preset: {key!r}")
    return user_id, version


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


def _neural_factory(
    model_uri: str,
    version: str,
    *,
    display_name: str | None = None,
) -> Callable[..., ChessBot]:
    def factory(
        *,
        temperature: float = 0.0,
        on_progress: Callable[[str], None] | None = None,
    ) -> ChessBot:
        from chess_teacher.bots.neural_baseline_bot import NeuralBaselineBot

        return NeuralBaselineBot(
            model_uri=model_uri,
            version=version,
            display_name=display_name,
            temperature=temperature,
            on_progress=on_progress,
        )

    return factory


def _baseline_factory(model_uri: str, version: str) -> Callable[..., ChessBot]:
    return _neural_factory(model_uri, version)


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


def invalidate_baseline_presets_cache(*, db_client: DatabaseClient | None = None) -> None:
    """Drop cached baseline and personal presets (all clients, or one client)."""
    if db_client is None:
        _baseline_presets_cache.clear()
        _personal_presets_cache.clear()
        return
    _baseline_presets_cache.pop(id(db_client), None)
    stale = [key for key in _personal_presets_cache if key[0] == id(db_client)]
    for key in stale:
        _personal_presets_cache.pop(key, None)


def reset_baseline_presets_cache_for_tests() -> None:
    """Test helper: clear baseline and personal preset caches between cases."""
    invalidate_baseline_presets_cache()


def _fetch_baseline_presets_from_db(db_client: DatabaseClient) -> list[BotPreset]:
    """Load playable baseline presets from Postgres (uncached)."""
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


def list_baseline_presets(
    db_client: DatabaseClient,
    *,
    force_refresh: bool = False,
) -> list[BotPreset]:
    """Playable baseline policies: production + archived (once-promoted) only.

    Results are cached in-process for ``_BASELINE_PRESETS_CACHE_TTL_SEC`` to avoid
    hitting Postgres on every Streamlit rerun during an active game.
    """
    cache_key = id(db_client)
    now = time.monotonic()
    if not force_refresh:
        cached = _baseline_presets_cache.get(cache_key)
        if cached is not None:
            presets, cached_at = cached
            if now - cached_at < _BASELINE_PRESETS_CACHE_TTL_SEC:
                return list(presets)

    presets = _fetch_baseline_presets_from_db(db_client)
    _baseline_presets_cache[cache_key] = (presets, now)
    return list(presets)


def _personal_preset(row: PersonalModel) -> BotPreset | None:
    from chess_teacher.pipelines.neural_network.models import BaselineModelStatus

    model_uri = row.model_uri
    if model_uri is None or not row.looks_like_candidate_style():
        return None
    if row.status not in {BaselineModelStatus.PRODUCTION, BaselineModelStatus.ARCHIVED}:
        return None
    status_label = "current" if row.status == BaselineModelStatus.PRODUCTION else "archived"
    return BotPreset(
        key=personal_preset_key(row.user_id, row.version),
        label=f"Personal {row.version}",
        description=f"Personalized on your games ({status_label}).",
        factory=_neural_factory(
            model_uri,
            row.version,
            display_name=f"Personal {row.version}",
        ),
    )


def _platform_fallback_preset(db_client: DatabaseClient) -> BotPreset | None:
    """Current platform production model, labeled as the personal-bot fallback."""
    from chess_teacher.pipelines.neural_network.models import (
        BaselineModel,
        BaselineModelStatus,
    )

    row = BaselineModel.latest_with_status(db_client, BaselineModelStatus.PRODUCTION)
    if row is None or not row.model_uri or not row.looks_like_candidate_style():
        return None
    return BotPreset(
        key=baseline_preset_key(row.version),
        label=f"Baseline {row.version}",
        description="Platform baseline. No personal model is promoted for you yet.",
        factory=_baseline_factory(row.model_uri, row.version),
    )


def list_personal_play_presets(
    db_client: DatabaseClient,
    user_id: str,
    *,
    force_refresh: bool = False,
) -> tuple[list[BotPreset], bool]:
    """Playable personal models for ``user_id``.

    Returns ``(presets, using_baseline_fallback)``. When this user has no
    promoted or archived personal model, the single option is the current
    platform production baseline and the flag is true.
    """
    cache_key = (id(db_client), user_id)
    now = time.monotonic()
    if not force_refresh:
        cached = _personal_presets_cache.get(cache_key)
        if cached is not None:
            presets, using_fallback, cached_at = cached
            if now - cached_at < _BASELINE_PRESETS_CACHE_TTL_SEC:
                return list(presets), using_fallback

    from chess_teacher.pipelines.neural_network.models import (
        BaselineModelStatus,
        PersonalModel,
    )

    rows = [
        row
        for row in PersonalModel.rows_for_user(db_client, user_id)
        if _personal_preset(row) is not None
    ]
    rows.sort(
        key=lambda row: (
            0 if row.status == BaselineModelStatus.PRODUCTION else 1,
            -(row.trained_at.timestamp() if row.trained_at is not None else 0.0),
        )
    )
    presets = [preset for row in rows if (preset := _personal_preset(row)) is not None]
    using_fallback = False
    if not presets:
        fallback = _platform_fallback_preset(db_client)
        if fallback is not None:
            presets = [fallback]
            using_fallback = True

    _personal_presets_cache[cache_key] = (presets, using_fallback, now)
    return list(presets), using_fallback


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


def get_bot_preset(
    key: str,
    *,
    db_client: DatabaseClient | None = None,
    user_id: str | None = None,
) -> BotPreset:
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
        by_key = {preset.key: preset for preset in list_baseline_presets(client)}
        preset = by_key.get(key)
        if preset is not None:
            return preset
        raise KeyError(f"Unknown baseline bot preset: {key!r} (version={version!r})")

    if key.startswith(PERSONAL_PRESET_PREFIX):
        owner_id, version = parse_personal_preset_key(key)
        if user_id is not None and owner_id != user_id:
            raise KeyError(f"Personal bot preset {key!r} is not for this user")
        client = db_client
        if client is None:
            from chess_teacher.utils.db.client import get_db_client

            client = get_db_client()
        presets, _using_fallback = list_personal_play_presets(client, owner_id)
        preset = next((item for item in presets if item.key == key), None)
        if preset is not None:
            return preset
        raise KeyError(f"Unknown personal bot preset: {key!r} (version={version!r})")

    raise KeyError(f"Unknown bot preset: {key!r}")


def category_for_preset_key(key: str) -> OpponentCategory:
    """Best-effort category for restoring setup UI from a preset key."""
    if key.startswith(STOCKFISH_PRESET_PREFIX) or key in _LEGACY_STOCKFISH:
        return OpponentCategory.STOCKFISH
    if key.startswith(BASELINE_PRESET_PREFIX):
        return OpponentCategory.BASELINE
    if key.startswith(PERSONAL_PRESET_PREFIX):
        return OpponentCategory.PERSONAL
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
