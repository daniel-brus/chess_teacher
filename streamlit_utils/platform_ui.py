"""Reusable platform logos + account rows for Streamlit (any widget layout)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from typing import Literal

import streamlit as st

from chess_teacher.platform.account import Account, AccountPlatform

_DEFAULT_ICON_FRACTION = 0.08


def render_platform_logo(platform: AccountPlatform, *, width: int = 22) -> None:
    """Draw platform SVG when ``logo_path()`` exists under ``RAW_DIR``."""
    logo = platform.logo_path()
    if logo.is_file():
        st.image(str(logo), width=width)


@contextmanager
def platform_row(
    platform: AccountPlatform,
    *,
    logo_width: int = 22,
    icon_fraction: float = _DEFAULT_ICON_FRACTION,
) -> Iterator[AccountPlatform]:
    """Icon column + content column; ``yield`` platform for labels/widgets."""
    icon_col, content_col = st.columns(
        [icon_fraction, 1 - icon_fraction],
        vertical_alignment="center",
    )
    with icon_col:
        render_platform_logo(platform, width=logo_width)
    with content_col:
        yield platform


@contextmanager
def account_row(
    account: Account,
    *,
    logo_width: int = 22,
    icon_fraction: float = _DEFAULT_ICON_FRACTION,
) -> Iterator[Account]:
    """Icon column + content column; ``yield`` account for labels/widgets."""
    icon_col, content_col = st.columns(
        [icon_fraction, 1 - icon_fraction],
        vertical_alignment="center",
    )
    with icon_col:
        render_platform_logo(account.platform, width=logo_width)
    with content_col:
        yield account


class AccountPickMode(StrEnum):
    MULTI = "multi"
    SINGLE = "single"


def _resolve_selected_account(
    accounts: list[Account],
    *,
    key_prefix: str,
    default: Account | None,
) -> Account:
    state_key = f"{key_prefix}_selected_id"
    valid_ids = {account.account_id for account in accounts}
    current = st.session_state.get(state_key)
    if current not in valid_ids:
        chosen = default or accounts[0]
        st.session_state[state_key] = chosen.account_id
    return next(
        account for account in accounts if account.account_id == st.session_state[state_key]
    )


def pick_one_account(
    accounts: list[Account],
    *,
    label: str = "Account",
    key_prefix: str = "account_picker",
    default: Account | None = None,
    disabled: bool = False,
    show_logo: bool = True,
    logo_width: int = 24,
    change_label: str = "Change",
    allow_change: bool = True,
) -> Account | None:
    """Single selection: platform icon beside username; ``Change`` popover to switch.

    Set ``allow_change=False`` to freeze the display (e.g. while a pipeline runs).
    Use outside ``st.form`` (popover uses buttons). Pair with ``form_submit_button``.
    """
    if not accounts:
        return None
    if disabled:
        if label:
            st.markdown(f"**{label}**")
        st.caption("No accounts available.")
        return None

    selected = _resolve_selected_account(accounts, key_prefix=key_prefix, default=default)

    if label:
        st.markdown(f"**{label}**")

    icon_col, text_col, action_col = st.columns([0.07, 0.68, 0.25], vertical_alignment="center")
    with icon_col:
        if show_logo:
            render_platform_logo(selected.platform, width=logo_width)
    with text_col:
        st.markdown(f"**{selected.username}**")
    with action_col:
        if allow_change:
            with st.popover(change_label):
                for account in accounts:
                    opt_icon, opt_btn = st.columns([0.14, 0.86], vertical_alignment="center")
                    with opt_icon:
                        if show_logo:
                            render_platform_logo(account.platform, width=20)
                    with opt_btn:
                        is_current = account.account_id == selected.account_id
                        if st.button(
                            f"{account.username}{' ✓' if is_current else ''}",
                            key=f"{key_prefix}_pick_{account.account_id}",
                            use_container_width=True,
                            help=account.format_label(),
                        ):
                            st.session_state[f"{key_prefix}_selected_id"] = account.account_id
                            st.rerun()
        else:
            st.button(change_label, disabled=True, use_container_width=True)

    return _resolve_selected_account(accounts, key_prefix=key_prefix, default=default)


def pick_accounts_multi(
    accounts: list[Account],
    *,
    label: str = "Account",
    key_prefix: str = "account_picker",
    default_checked: bool = True,
    warn_if_empty: bool = True,
) -> list[Account]:
    """Multi selection via logo + checkboxes."""
    if label:
        st.markdown(f"**{label}**")
    selected: list[Account] = []
    for account in accounts:
        with account_row(account) as row_account:
            if st.checkbox(
                row_account.username,
                value=default_checked,
                key=f"{key_prefix}_{row_account.account_id}",
                help=row_account.format_label(),
            ):
                selected.append(row_account)
    if warn_if_empty and accounts and not selected:
        st.warning("Select at least one account.")
    return selected


def pick_accounts(
    accounts: list[Account],
    *,
    mode: AccountPickMode | Literal["multi", "single"] = AccountPickMode.MULTI,
    label: str = "Account",
    key_prefix: str = "account_picker",
    default_checked: bool = True,
    default_single: Account | None = None,
    disabled: bool = False,
    warn_if_empty: bool = True,
) -> list[Account] | Account | None:
    """Unified account picker — ``multi`` (checkboxes) or ``single`` (popover + logo)."""
    pick_mode = AccountPickMode(mode)
    if pick_mode == AccountPickMode.SINGLE:
        return pick_one_account(
            accounts,
            label=label,
            key_prefix=key_prefix,
            default=default_single,
            disabled=disabled,
        )
    return pick_accounts_multi(
        accounts,
        label=label,
        key_prefix=key_prefix,
        default_checked=default_checked,
        warn_if_empty=warn_if_empty,
    )


def pick_platform(
    *,
    label: str = "Platform",
    key_prefix: str = "platform_picker",
    platforms: list[AccountPlatform] | None = None,
    horizontal: bool = True,
    default: AccountPlatform | None = None,
) -> AccountPlatform:
    """Platform choice with logos (works inside ``st.form`` via ``st.radio``)."""
    options = platforms if platforms is not None else list(AccountPlatform)
    if label:
        st.markdown(f"**{label}**")

    logo_cols = st.columns(len(options))
    for column, platform in zip(logo_cols, options, strict=True):
        with column:
            render_platform_logo(platform, width=28)

    index = 0
    if default is not None and default in options:
        index = options.index(default)

    return st.radio(
        label,
        options,
        index=index,
        format_func=lambda platform: platform.value,
        horizontal=horizontal,
        key=key_prefix,
        label_visibility="collapsed",
    )
