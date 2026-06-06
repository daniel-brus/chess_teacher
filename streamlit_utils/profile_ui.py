"""Sidebar profile avatar + settings upload helpers."""

from __future__ import annotations

import html

import streamlit as st

from chess_teacher.platform.profile_picture import profile_pictures
from chess_teacher.platform.user import User
from streamlit_utils.layout import ingest_css

# SVG avatars (app logos, etc.): opaque circle behind transparent wordmark areas.
_AVATAR_SVG_CIRCLE_BG = "#808080"


def render_profile_avatar(
    user: User,
    *,
    size_px: int = 88,
    centered: bool = True,
) -> None:
    """Circular avatar from OAuth URL or uploaded file."""
    alt = html.escape(user.name or user.email or "Profile")
    src = profile_pictures.picture_img_src(user.picture)
    if src is None:
        ingest_css(
            f'<div style="text-align:center;font-size:{size_px}px;line-height:1;">♟️</div>',
        )
        return

    if src.startswith("data:"):
        img_src = src
    else:
        img_src = html.escape(src, quote=True)

    if profile_pictures.is_svg_picture(user.picture):
        circle = (
            f"width:{size_px}px;height:{size_px}px;border-radius:50%;"
            f"background:{_AVATAR_SVG_CIRCLE_BG};overflow:hidden;"
            f"display:flex;align-items:center;justify-content:center;flex-shrink:0;"
        )
        img = (
            f'<div style="{circle}">'
            f'<img src="{img_src}" alt="{alt}" '
            f'style="display:block;object-fit:contain;width:85%;height:85%;'
            f'pointer-events:none;user-select:none;">'
            f"</div>"
        )
    else:
        img = (
            f'<img src="{img_src}" alt="{alt}" width="{size_px}" height="{size_px}" '
            f'style="display:block;border-radius:50%;object-fit:cover;'
            f'width:{size_px}px;height:{size_px}px;pointer-events:none;user-select:none;">'
        )
    if centered:
        st.html(
            f'<div style="display:flex;justify-content:center;margin-bottom:0.75rem;">{img}</div>'
        )
    else:
        st.html(img)


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


def profile_picture_preview(user: User, *, size_px: int = 120) -> None:
    """Settings page preview of the current avatar."""
    st.caption("Current profile picture")
    render_profile_avatar(user, size_px=size_px, centered=False)
