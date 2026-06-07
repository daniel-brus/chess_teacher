"""Sidebar profile avatar + settings upload helpers."""

from __future__ import annotations

import html

import streamlit as st

from chess_teacher.platform.account import AppLogoVariant
from chess_teacher.platform.profile_picture import profile_pictures
from chess_teacher.platform.user import User
from streamlit_utils.layout import ingest_css

# SVG avatars (app logos, etc.): opaque circle behind transparent wordmark areas.
_AVATAR_SVG_CIRCLE_BG = "#808080"
_ACTIVE_PRESET_OUTLINE = "#2563eb"


def _avatar_markup(
    *,
    img_src: str | None,
    is_svg: bool,
    size_px: int,
    alt: str,
) -> str:
    if img_src is None:
        return f'<div style="text-align:center;font-size:{size_px}px;line-height:1;">♟️</div>'

    if img_src.startswith("data:"):
        escaped_src = img_src
    else:
        escaped_src = html.escape(img_src, quote=True)

    if is_svg:
        circle = (
            f"width:{size_px}px;height:{size_px}px;border-radius:50%;"
            f"background:{_AVATAR_SVG_CIRCLE_BG};overflow:hidden;"
            f"display:flex;align-items:center;justify-content:center;flex-shrink:0;"
        )
        return (
            f'<div style="{circle}">'
            f'<img src="{escaped_src}" alt="{html.escape(alt)}" '
            f'style="display:block;object-fit:contain;width:85%;height:85%;'
            f'pointer-events:none;user-select:none;">'
            f"</div>"
        )

    return (
        f'<img src="{escaped_src}" alt="{html.escape(alt)}" width="{size_px}" height="{size_px}" '
        f'style="display:block;border-radius:50%;object-fit:cover;'
        f'width:{size_px}px;height:{size_px}px;pointer-events:none;user-select:none;">'
    )


def render_avatar_preview(
    *,
    img_src: str | None,
    is_svg: bool = False,
    size_px: int = 120,
    centered: bool = True,
    alt: str = "Preview",
) -> None:
    """Circular avatar preview from an image source (settings dialogs and presets)."""
    markup = _avatar_markup(img_src=img_src, is_svg=is_svg, size_px=size_px, alt=alt)
    if centered:
        st.html(f'<div style="display:flex;justify-content:center;margin:0.5rem 0;">{markup}</div>')
    else:
        st.html(markup)


def highlight_active_preset(container_key: str) -> None:
    """Blue ring around a preset card without changing its layout size."""
    ingest_css(
        f"""
div[class*="st-key-{container_key}"] {{
    outline: 2px solid {_ACTIVE_PRESET_OUTLINE};
    outline-offset: 2px;
    border-radius: 0.5rem;
}}
"""
    )


def render_logo_preset_preview(*, variant: AppLogoVariant, size_px: int = 88) -> None:
    """Sidebar-style preview of a black or white app-logo avatar."""
    picture_ref = profile_pictures.app_logo_picture_ref(variant=variant)
    render_avatar_preview(
        img_src=profile_pictures.picture_img_src(picture_ref),
        is_svg=True,
        size_px=size_px,
        alt=f"{variant} app logo",
    )


def render_profile_avatar(
    user: User,
    *,
    size_px: int = 88,
    centered: bool = True,
) -> None:
    """Circular avatar from OAuth URL or uploaded file."""
    alt = user.name or user.email or "Profile"
    src = profile_pictures.picture_img_src(user.picture)
    markup = _avatar_markup(
        img_src=src,
        is_svg=profile_pictures.is_svg_picture(user.picture),
        size_px=size_px,
        alt=alt,
    )
    if centered:
        st.html(
            f'<div style="display:flex;justify-content:center;margin-bottom:0.75rem;">{markup}</div>'
        )
    else:
        st.html(markup)


def render_sidebar_profile(user: User) -> None:
    """Profile block at top of sidebar (avatar + display name)."""
    ingest_css(
        """
<style>
div[class*="st-key-sidebar_profile"] {
    margin-bottom: 0.5rem;
}
</style>
"""
    )
    with st.container(key="sidebar_profile"):
        render_profile_avatar(user)
        display_name = user.name or user.given_name or user.email
        if display_name:
            ingest_css(
                f'<p style="text-align:center;margin:0 0 0.5rem;font-weight:600;">'
                f"{html.escape(display_name)}</p>",
            )
        st.divider()
