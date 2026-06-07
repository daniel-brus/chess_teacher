from typing import Literal

import streamlit as st

from chess_teacher.platform.account import Account
from chess_teacher.platform.profile_picture import (
    clear_upload_image_cache,
    profile_pictures,
    replace_user_profile_picture,
    replace_user_profile_picture_with_app_logo,
)
from chess_teacher.platform.users_accounts import (
    add_account,
    get_accounts_for_user,
    remove_account,
    remove_all_accounts_for_user,
)
from chess_teacher.utils.db_client import get_db_client
from chess_teacher.utils.logging_utils import get_logger
from streamlit_utils.login import require_authenticated_user
from streamlit_utils.page_config import configure_page
from streamlit_utils.platform_ui import pick_platform, render_platform_logo
from streamlit_utils.profile_ui import profile_picture_preview
from streamlit_utils.session_state import force_logout, set_current_user
from streamlit_utils.theme import (
    active_appearance,
    default_pill_value,
    theme_choice_options,
    theme_option_markdown,
)

configure_page("Settings")

user = require_authenticated_user()

db_client = get_db_client()
logger = get_logger()

_MAX_PROFILE_PICTURE_SIZE_MB = 10
_PROFILE_UPLOAD_NONCE_KEY = "settings_profile_picture_upload_nonce"


def _profile_picture_upload_key() -> str:
    nonce = st.session_state.setdefault(_PROFILE_UPLOAD_NONCE_KEY, 0)
    return f"settings_profile_picture_upload_{nonce}"


def _clear_profile_picture_uploader() -> None:
    upload_key = _profile_picture_upload_key()
    st.session_state.pop(upload_key, None)
    st.session_state[_PROFILE_UPLOAD_NONCE_KEY] = st.session_state[_PROFILE_UPLOAD_NONCE_KEY] + 1


st.title("Personal Settings")


def _show_profile_picture_form() -> None:
    profile_picture_preview(user)

    with st.form("profile_picture_form"):
        upload = st.file_uploader(
            "Upload a new profile picture",
            max_upload_size=_MAX_PROFILE_PICTURE_SIZE_MB,
            type=["png", "jpg", "jpeg", "webp", "gif"],
            help="Replaces your current sidebar avatar. PNG, JPEG, WebP, or GIF.",
            key=_profile_picture_upload_key(),
        )
        submitted = st.form_submit_button("Save profile picture")

    if not submitted:
        return

    if upload is None:
        st.warning("Choose an image to upload.")
        return

    try:
        updated_user = replace_user_profile_picture(
            user,
            db_client,
            data=upload.getvalue(),
            original_filename=upload.name,
        )
    except ValueError as e:
        logger.log_and_raise(e, "Failed to save profile picture")

    set_current_user(updated_user)
    _clear_profile_picture_uploader()
    st.success("Profile picture saved.")
    st.rerun()


def _set_profile_picture_to_app_logo(variant: Literal["black", "white"]) -> None:
    try:
        updated_user = replace_user_profile_picture_with_app_logo(
            user,
            db_client,
            variant=variant,
        )
    except (ValueError, FileNotFoundError) as e:
        logger.log_and_raise(e, "Failed to set app logo profile picture")

    set_current_user(updated_user)
    label = "black" if variant == "black" else "white"
    st.success(f"Profile picture set to the {label} app logo.")
    st.rerun()


def _reset_profile_picture_to_provider() -> None:
    provider_picture = st.user.to_dict().get("picture")
    if not provider_picture:
        st.info("Your sign-in provider did not supply a profile picture.")
        return

    clear_upload_image_cache(user.picture)
    profile_pictures.delete(user.picture)
    user.upsert_field(db_client, "picture", provider_picture)
    user.picture = provider_picture
    set_current_user(user)
    st.success("Profile picture reset to your sign-in provider photo.")
    st.rerun()


def _show_add_account_form() -> None:
    with st.form("add_platform_account"):
        platform = pick_platform(key_prefix="settings_add_platform")
        username = st.text_input("Username")
        submitted = st.form_submit_button("Add account")

    if not submitted:
        return

    username = username.strip()
    if not username:
        st.warning("Enter a username.")
        return

    account = Account.from_username_and_platform(username=username, platform=platform)
    added = add_account(user, account, db_client)
    if added:
        st.success(f"{platform.value}-account added.")
    else:
        st.info("This account is already linked to your user.")
    st.session_state["show_add_platform_account"] = False
    st.rerun()


def _show_account_list(accounts_list: list[Account]) -> None:
    header_cols = st.columns([2, 3, 2, 2])
    header_cols[0].markdown("**Platform**")
    header_cols[1].markdown("**Username**")
    header_cols[2].markdown("**Latest ingestion**")
    header_cols[3].markdown("**Remove**")

    for account in accounts_list:
        cols = st.columns([2, 3, 2, 2])
        with cols[0]:
            render_platform_logo(account.platform, width=24)
        cols[1].write(account.username)
        cols[2].write(account.latest_ingestion or "Never")
        if cols[3].button("Remove", key=f"remove_{account.account_id}"):
            remove_account(user, account, db_client)
            st.success("Account unlinked.")
            st.rerun()


@st.dialog("Are you sure?")
def _safe_remove_user():
    st.warning("Your user information will be lost forever")
    if st.button("I'm sure"):
        remove_all_accounts_for_user(user, db_client)
        user.delete_from_db(db_client)
        force_logout()


st.subheader("Profile picture")

_show_profile_picture_form()

st.caption("Or use the Chess Teacher logo")
_logo_black_col, _logo_white_col = st.columns(2)
with _logo_black_col:
    if st.button("Black logo", key="settings_profile_black_logo", width="stretch"):
        _set_profile_picture_to_app_logo("black")
with _logo_white_col:
    if st.button("White logo", key="settings_profile_white_logo", width="stretch"):
        _set_profile_picture_to_app_logo("white")

_provider_picture = st.user.to_dict().get("picture")
if user.picture != _provider_picture and profile_pictures.is_remote(_provider_picture):
    if st.button("Use sign-in provider photo"):
        _reset_profile_picture_to_provider()

st.divider()

st.subheader("Linked platform accounts")

accounts = get_accounts_for_user(user, db_client)
if accounts:
    _show_account_list(accounts)
else:
    st.info("There are no platform accounts linked.")

if st.button("Add new platform account"):
    st.session_state["show_add_platform_account"] = True

if st.session_state.get("show_add_platform_account", False):
    _show_add_account_form()

st.divider()

st.subheader("Appearance")

_light_ids = [option[0] for option in theme_choice_options("light")]
_dark_ids = [option[0] for option in theme_choice_options("dark")]
_light_default = default_pill_value(user.default_light_theme_id, _light_ids)
_dark_default = default_pill_value(user.default_dark_theme_id, _dark_ids)
_active = active_appearance()

with st.form("default_theme_form"):
    st.caption(
        "Chessboard background and dividers. The active set follows the Streamlit "
        "⋮ menu (Light/Dark). Widget colors still come from that menu."
    )
    if _active == "light":
        st.info("Streamlit is in **light** mode — the app uses your light theme until you switch.")
    else:
        st.info("Streamlit is in **dark** mode — the app uses your dark theme until you switch.")

    selected_light_id = st.pills(
        "Light mode background",
        options=_light_ids,
        default=_light_default,
        format_func=lambda theme_id: theme_option_markdown(theme_id, "light"),
        selection_mode="single",
        help="Used when Streamlit appearance is Light.",
    )
    selected_dark_id = st.pills(
        "Dark mode background",
        options=_dark_ids,
        default=_dark_default,
        format_func=lambda theme_id: theme_option_markdown(theme_id, "dark"),
        selection_mode="single",
        help="Used when Streamlit appearance is Dark.",
    )
    if selected_light_id is None:
        selected_light_id = _light_default
    if selected_dark_id is None:
        selected_dark_id = _dark_default
    if st.form_submit_button("Save appearance"):
        user.upsert_field(db_client, "default_light_theme_id", selected_light_id or None)
        user.upsert_field(db_client, "default_dark_theme_id", selected_dark_id or None)
        user.default_light_theme_id = selected_light_id or None
        user.default_dark_theme_id = selected_dark_id or None
        set_current_user(user)
        st.success("Appearance saved.")
        st.rerun()

st.divider()

if st.button("Remove user"):
    _safe_remove_user()
