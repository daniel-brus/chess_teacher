"""App theme registry: chessboard background and dividers per Streamlit appearance.

Each appearance (light / dark menu) has its own named theme set in ``themes.toml``.
Users store a default theme id per appearance on their profile.
"""

from __future__ import annotations

import base64
import html
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import streamlit as st

from chess_teacher.utils.logging import get_logger
from streamlit_utils.bg_icons import generate_chessboard_svg

if TYPE_CHECKING:
    from chess_teacher.platform.user import User

Appearance = Literal["light", "dark"]
SESSION_PALETTE_KEY = "_ct_theme_palette"
_THEMES_PATH = Path(__file__).resolve().parent.parent / ".streamlit" / "themes.toml"
# Empty id = built-in generic palette (``st.pills`` options are strings).
_BUILTIN_DEFAULT = ""
_BUILTIN_LABEL = "Built-in default"
DIVIDER_WIDTH_PX = 2
_GLOBAL_APP_CSS_INJECTED_KEY = "_ct_global_app_css_injected"
_LAST_INJECTED_PALETTE_KEY = "_ct_last_injected_palette"
_theme_registry_cache: (
    tuple[
        dict[Appearance, ThemePalette],
        dict[Appearance, dict[str, ThemeDefinition]],
    ]
    | None
) = None
_theme_registry_mtime: float | None = None


@dataclass(frozen=True, slots=True)
class ThemePalette:
    """Colors for chessboard background tiles and custom dividers."""

    tile_light: str
    tile_dark: str
    divider: str
    label: str = ""


@dataclass(frozen=True, slots=True)
class ThemeDefinition:
    theme_id: str
    appearance: Appearance
    palette: ThemePalette


def active_appearance() -> Appearance:
    """Resolved Streamlit appearance (system menu → light or dark)."""
    mode = st.context.theme.type
    return "dark" if mode == "dark" else "light"


def reset_theme_css_state() -> None:
    """Clear CSS injection flags (e.g. on entry script ``configure_page``)."""
    st.session_state.pop(_GLOBAL_APP_CSS_INJECTED_KEY, None)
    st.session_state.pop(_LAST_INJECTED_PALETTE_KEY, None)


def _load_theme_registry() -> tuple[
    dict[Appearance, ThemePalette],
    dict[Appearance, dict[str, ThemeDefinition]],
]:
    global _theme_registry_cache, _theme_registry_mtime

    if not _THEMES_PATH.is_file():
        msg = f"Missing theme config: {_THEMES_PATH}"
        raise FileNotFoundError(msg)

    mtime = _THEMES_PATH.stat().st_mtime
    if _theme_registry_cache is not None and _theme_registry_mtime == mtime:
        return _theme_registry_cache

    raw = tomllib.loads(_THEMES_PATH.read_text(encoding="utf-8"))
    generic: dict[Appearance, ThemePalette] = {}
    for mode in ("light", "dark"):
        section = raw.get("generic", {}).get(mode)
        if not isinstance(section, dict):
            msg = f"themes.toml must define [generic.{mode}]"
            raise ValueError(msg)
        generic[mode] = _palette_from_toml(section)

    named: dict[Appearance, dict[str, ThemeDefinition]] = {"light": {}, "dark": {}}
    themes_root = raw.get("themes")
    if themes_root is not None:
        if not isinstance(themes_root, dict):
            msg = "themes.toml [themes] must be a table"
            raise ValueError(msg)
        for appearance in ("light", "dark"):
            bucket = themes_root.get(appearance, {})
            if not isinstance(bucket, dict):
                msg = f"themes.toml [themes.{appearance}] must be a table"
                raise ValueError(msg)
            for theme_id, section in bucket.items():
                if not isinstance(section, dict):
                    msg = f"themes.toml [themes.{appearance}.{theme_id}] must be a table"
                    raise ValueError(msg)
                named[appearance][theme_id] = ThemeDefinition(
                    theme_id=theme_id,
                    appearance=appearance,
                    palette=_palette_from_toml(section),
                )

    _theme_registry_cache = (generic, named)
    _theme_registry_mtime = mtime
    return _theme_registry_cache


def _palette_from_toml(section: dict[str, object]) -> ThemePalette:
    try:
        return ThemePalette(
            tile_light=str(section["tile_light"]),
            tile_dark=str(section["tile_dark"]),
            divider=str(section["divider"]),
            label=str(section.get("label", "")),
        )
    except KeyError as exc:
        msg = f"Theme section missing required key: {exc.args[0]}"
        raise ValueError(msg) from exc


def _display_label(palette: ThemePalette) -> str:
    return palette.label or _BUILTIN_LABEL


def generic_palette(appearance: Appearance | None = None) -> ThemePalette:
    generic, _ = _load_theme_registry()
    return generic[appearance or active_appearance()]


def get_named_theme(theme_id: str, appearance: Appearance) -> ThemeDefinition | None:
    _, named = _load_theme_registry()
    return named.get(appearance, {}).get(theme_id)


def list_named_themes(appearance: Appearance) -> list[ThemeDefinition]:
    _, named = _load_theme_registry()
    return sorted(named.get(appearance, {}).values(), key=lambda t: t.palette.label.lower())


def _escape_svg_color(color: str) -> str:
    return html.escape(color, quote=True)


def _swatch_svg_data_uri(palette: ThemePalette, *, width: int = 52, height: int = 16) -> str:
    """Tiny light / dark tiles + divider as a base64 SVG for Markdown image icons."""
    tile_light = _escape_svg_color(palette.tile_light)
    tile_dark = _escape_svg_color(palette.tile_dark)
    divider = _escape_svg_color(palette.divider)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'<rect x="1" y="1" width="14" height="14" fill="{tile_light}" '
        f'stroke="#888" stroke-width="0.5"/>'
        f'<rect x="17" y="1" width="14" height="14" fill="{tile_dark}"/>'
        f'<rect x="33" y="7" width="18" height="2" fill="{divider}"/>'
        "</svg>"
    )
    encoded = base64.b64encode(svg.encode()).decode()
    return f"data:image/svg+xml;base64,{encoded}"


def theme_option_markdown(theme_id: str, appearance: Appearance) -> str:
    """Markdown label for ``st.pills`` (swatch image + name)."""
    if theme_id == _BUILTIN_DEFAULT:
        palette = generic_palette(appearance)
        return f"![]({_swatch_svg_data_uri(palette)}) {_display_label(palette)}"
    theme = get_named_theme(theme_id, appearance)
    if theme is None:
        return theme_id
    label = theme.palette.label or theme_id
    return f"![]({_swatch_svg_data_uri(theme.palette)}) {label}"


def user_default_theme_id(user: User | None, appearance: Appearance) -> str | None:
    """Saved theme id for the given appearance, or None for built-in generic palette."""
    if user is None:
        return None
    if appearance == "light":
        return user.default_light_theme_id
    return user.default_dark_theme_id


def resolve_theme(
    user: User | None = None, *, appearance: Appearance | None = None
) -> ThemeDefinition | None:
    """Named theme for active (or given) appearance from the user's profile."""
    mode = appearance or active_appearance()
    theme_id = user_default_theme_id(user, mode)
    if not theme_id:
        return None
    theme = get_named_theme(theme_id, mode)
    if theme is None:
        get_logger().warning(
            "Unknown %s theme id %r for user — falling back to built-in palette",
            mode,
            theme_id,
        )
    return theme


def resolve_palette(
    user: User | None = None, *, appearance: Appearance | None = None
) -> ThemePalette:
    """User default for current appearance, else built-in generic palette."""
    mode = appearance or active_appearance()
    named = resolve_theme(user, appearance=mode)
    if named is not None:
        return named.palette
    return generic_palette(mode)


def current_palette() -> ThemePalette:
    stored = st.session_state.get(SESSION_PALETTE_KEY)
    if isinstance(stored, ThemePalette):
        return stored
    return generic_palette()


def divider_rgba() -> str:
    return current_palette().divider


def divider_width_px() -> int:
    return DIVIDER_WIDTH_PX


def divider_border() -> str:
    return f"{DIVIDER_WIDTH_PX}px solid {divider_rgba()}"


def _chessboard_background_css(color1: str, color2: str, *, square_size: int = 40) -> str:
    svg, tile_width, tile_height = generate_chessboard_svg(color1, color2, square_size=square_size)
    encoded = base64.b64encode(svg.encode()).decode()
    return f"""
.stApp {{
    background-image: url("data:image/svg+xml;base64,{encoded}");
    background-repeat: repeat;
    background-size: {tile_width}px {tile_height}px;
}}
.stAppHeader {{
    background-color: transparent !important;
}}
"""


def _themed_dividers_css(palette: ThemePalette) -> str:
    rule = f"{DIVIDER_WIDTH_PX}px solid {palette.divider}"
    return f"""
.stApp hr {{
    border: none;
    border-top: {rule};
    background-color: transparent;
    height: 0;
    opacity: 1;
}}
"""


def palette_app_css(palette: ThemePalette) -> str:
    """Chessboard background and themed horizontal rules."""
    return _chessboard_background_css(palette.tile_light, palette.tile_dark) + _themed_dividers_css(
        palette
    )


def apply_app_theme(user: User | None = None) -> ThemePalette:
    """Apply global app styles once per rerun (re-injected when the page DOM is rebuilt)."""
    from streamlit_utils.layout import ingest_css, shell_css

    palette = resolve_palette(user)
    st.session_state[SESSION_PALETTE_KEY] = palette
    palette_sig = (palette.tile_light, palette.tile_dark, palette.divider)
    if (
        st.session_state.get(_GLOBAL_APP_CSS_INJECTED_KEY)
        and st.session_state.get(_LAST_INJECTED_PALETTE_KEY) == palette_sig
    ):
        return palette

    st.session_state[_GLOBAL_APP_CSS_INJECTED_KEY] = True
    st.session_state[_LAST_INJECTED_PALETTE_KEY] = palette_sig

    ingest_css(shell_css() + palette_app_css(palette))
    return palette


def theme_choice_options(appearance: Appearance) -> list[tuple[str, str]]:
    """Pill options: (theme_id, label). Empty id = built-in generic palette."""
    generic = generic_palette(appearance)
    options: list[tuple[str, str]] = [
        (_BUILTIN_DEFAULT, _display_label(generic)),
    ]
    for theme in list_named_themes(appearance):
        options.append((theme.theme_id, theme.palette.label or theme.theme_id))
    return options


def default_pill_value(saved_id: str | None, option_ids: list[str]) -> str:
    if saved_id and saved_id in option_ids:
        return saved_id
    return option_ids[0] if option_ids else _BUILTIN_DEFAULT
