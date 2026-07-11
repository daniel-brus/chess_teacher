import base64
import mimetypes
from datetime import datetime
from typing import Any, Literal
from zoneinfo import available_timezones

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from chess_teacher.platform.account import Account
from chess_teacher.platform.profile_picture import (
    clear_upload_image_cache,
    profile_pictures,
)
from chess_teacher.platform.user import (
    cron_time_option_index,
    dispatch_cron_time_options,
    format_cron_time_label,
)
from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.general_utils import assert_valid_timezone
from chess_teacher.utils.logging import get_logger
from streamlit_utils.login import require_authenticated_user
from streamlit_utils.page_config import configure_page
from streamlit_utils.page_logging import log_page_view, log_user_action
from streamlit_utils.platform_ui import pick_platform, render_platform_logo
from streamlit_utils.profile_ui import (
    highlight_active_preset,
    render_avatar_preview,
    render_logo_preset_preview,
)
from streamlit_utils.session_state import force_logout, set_current_user
from streamlit_utils.theme import (
    active_appearance,
    default_pill_value,
    theme_choice_options,
    theme_option_markdown,
)

configure_page("Settings")

user = require_authenticated_user()
log_page_view("Settings", user)

db_client = get_db_client()
logger = get_logger()

_MAX_PROFILE_PICTURE_SIZE_MB = 10
_MAX_PROFILE_PICTURE_SIZE_BYTES = _MAX_PROFILE_PICTURE_SIZE_MB * 1024 * 1024
_PROFILE_UPLOAD_NONCE_KEY = "settings_profile_picture_upload_nonce"
_PROFILE_PENDING_DIALOG_KEY = "settings_profile_pending_dialog"
_PROFILE_PENDING_UPLOAD_SIG_KEY = "settings_profile_pending_upload_sig"
_PROFILE_ALLOWED_UPLOAD_TYPES = ["png", "jpg", "jpeg", "webp", "gif"]
ProfilePictureDialogChoice = dict[str, Any]


def _profile_picture_upload_key() -> str:
    nonce = st.session_state.setdefault(_PROFILE_UPLOAD_NONCE_KEY, 0)
    return f"settings_profile_picture_upload_{nonce}"


def _clear_profile_picture_uploader() -> None:
    upload_key = _profile_picture_upload_key()
    st.session_state.pop(upload_key, None)
    st.session_state[_PROFILE_UPLOAD_NONCE_KEY] = st.session_state[_PROFILE_UPLOAD_NONCE_KEY] + 1


def _clear_profile_picture_dialog() -> None:
    st.session_state.pop(_PROFILE_PENDING_DIALOG_KEY, None)
    st.session_state.pop(_PROFILE_PENDING_UPLOAD_SIG_KEY, None)


def _cancel_profile_picture_dialog() -> None:
    pending = st.session_state.get(_PROFILE_PENDING_DIALOG_KEY)
    _clear_profile_picture_dialog()
    if isinstance(pending, dict) and pending.get("kind") == "upload":
        _clear_profile_picture_uploader()


def _queue_profile_picture_dialog(choice: ProfilePictureDialogChoice) -> None:
    st.session_state[_PROFILE_PENDING_DIALOG_KEY] = choice


def _upload_data_uri(upload: UploadedFile) -> str:
    mime, _ = mimetypes.guess_type(upload.name or "")
    mime = mime or "application/octet-stream"
    encoded = base64.b64encode(upload.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _upload_is_svg(upload: UploadedFile) -> bool:
    return (upload.name or "").lower().endswith(".svg")


def _apply_profile_picture_choice(
    choice: ProfilePictureDialogChoice,
    *,
    upload: UploadedFile | None = None,
) -> None:
    kind = choice["kind"]
    if kind == "upload":
        if upload is None:
            raise ValueError("Upload dialog requires a file.")
        updated_user = user.replace_profile_picture(
            db_client,
            data=upload.getvalue(),
            original_filename=upload.name,
        )
        set_current_user(updated_user)
        log_user_action(
            "Profile picture saved from upload",
            updated_user,
            filename=upload.name,
            size_bytes=upload.size,
        )
        st.success("Profile picture saved.")
        return

    if kind == "logo":
        variant = choice["variant"]
        try:
            updated_user = user.replace_profile_picture_with_app_logo(
                db_client,
                variant=variant,
            )
        except (ValueError, FileNotFoundError) as e:
            logger.log_and_raise(e, "Failed to set app logo profile picture")
        set_current_user(updated_user)
        log_user_action(
            "Profile picture set to app logo",
            updated_user,
            variant=variant,
        )
        label = "black" if variant == "black" else "white"
        st.success(f"Profile picture set to the {label} app logo.")
        return

    if kind == "provider":
        provider_picture = st.user.to_dict().get("picture")
        if not provider_picture:
            st.info("Your sign-in provider did not supply a profile picture.")
            return
        clear_upload_image_cache(user.picture)
        profile_pictures.delete(user.picture)
        user.upsert_field(db_client, "picture", provider_picture)
        user.picture = provider_picture
        set_current_user(user)
        log_user_action("Profile picture reset to sign-in provider photo", user)
        st.success("Profile picture set to your sign-in provider photo.")
        return

    raise ValueError(f"Unknown profile picture choice: {kind!r}")


def _render_profile_picture_dialog_preview(
    choice: ProfilePictureDialogChoice,
    *,
    upload: UploadedFile | None = None,
) -> str:
    kind = choice["kind"]
    if kind == "upload":
        if upload is None:
            return "Save this image as your profile picture?"
        render_avatar_preview(
            img_src=_upload_data_uri(upload),
            is_svg=_upload_is_svg(upload),
            alt="Uploaded profile picture",
        )
        return "Save this image as your profile picture?"

    if kind == "logo":
        variant = choice["variant"]
        render_logo_preset_preview(variant=variant)
        label = "black" if variant == "black" else "white"
        return f"Use the {label} Chess Teacher logo as your profile picture?"

    if kind == "provider":
        provider_picture = st.user.to_dict().get("picture")
        render_avatar_preview(
            img_src=provider_picture,
            is_svg=profile_pictures.is_svg_picture(provider_picture),
            alt="Sign-in provider profile picture",
        )
        return "Use your sign-in provider photo as your profile picture?"

    return "Update your profile picture?"


@st.dialog("Profile picture preview")
def _profile_picture_dialog(upload: UploadedFile | None = None) -> None:
    choice = st.session_state.get(_PROFILE_PENDING_DIALOG_KEY)
    if not isinstance(choice, dict):
        return

    st.write(_render_profile_picture_dialog_preview(choice, upload=upload))

    save_col, cancel_col = st.columns(2)
    with save_col:
        _save_pad_left, save_btn_col, _save_pad_right = st.columns([1, 1.4, 1])
        with save_btn_col:
            if st.button("Save", type="primary", key="settings_profile_dialog_save"):
                try:
                    _apply_profile_picture_choice(choice, upload=upload)
                except ValueError as e:
                    logger.log_and_raise(e, "Failed to save profile picture")
                _clear_profile_picture_dialog()
                if choice.get("kind") == "upload":
                    _clear_profile_picture_uploader()
                st.rerun()
    with cancel_col:
        _cancel_pad_left, cancel_btn_col, _cancel_pad_right = st.columns([1, 1.4, 1])
        with cancel_btn_col:
            if st.button("Cancel", key="settings_profile_dialog_cancel"):
                _cancel_profile_picture_dialog()
                st.rerun()


def _logo_preset_is_active(variant: Literal["black", "white"]) -> bool:
    return user.picture == profile_pictures.app_logo_picture_ref(variant=variant)


def _provider_preset_is_active(provider_picture: str | None) -> bool:
    return user.picture == provider_picture


def _show_provider_profile_preset(provider_picture: str) -> None:
    active = _provider_preset_is_active(provider_picture)
    container_key = "settings_preset_provider"
    with st.container(key=container_key):
        if active:
            highlight_active_preset(container_key)
        st.markdown("**Sign-in provider**")
        render_avatar_preview(
            img_src=provider_picture,
            is_svg=profile_pictures.is_svg_picture(provider_picture),
            size_px=88,
            alt="Sign-in provider profile picture",
        )
        if st.button("Use", key="settings_profile_provider", width="stretch"):
            _queue_profile_picture_dialog({"kind": "provider"})
            st.rerun()


def _show_logo_profile_preset(variant: Literal["black", "white"]) -> None:
    active = _logo_preset_is_active(variant)
    label = "Black logo" if variant == "black" else "White logo"
    container_key = f"settings_preset_logo_{variant}"
    with st.container(key=container_key):
        if active:
            highlight_active_preset(container_key)
        st.markdown(f"**{label}**")
        render_logo_preset_preview(variant=variant, size_px=88)
        if st.button("Use", key=f"settings_profile_logo_{variant}", width="stretch"):
            _queue_profile_picture_dialog({"kind": "logo", "variant": variant})
            st.rerun()


def _show_profile_tab() -> None:
    with st.form("profile_name_form"):
        display_name = st.text_input("Display name", value=user.name or "")
        if st.form_submit_button("Save display name"):
            name_value = display_name.strip() or None
            if name_value != user.name:
                updated_user = user.update_name(db_client, name_value)
                set_current_user(updated_user)
                log_user_action("Display name updated", updated_user, name=name_value)
            st.success("Display name saved.")
            st.rerun()

    st.divider()
    st.caption("Your current avatar is shown in the sidebar.")

    upload = st.file_uploader(
        "Upload a new profile picture",
        max_upload_size=_MAX_PROFILE_PICTURE_SIZE_MB,
        type=_PROFILE_ALLOWED_UPLOAD_TYPES,
        help=(
            f"Opens a preview before saving. PNG, JPEG, WebP, or GIF, "
            f"up to {_MAX_PROFILE_PICTURE_SIZE_MB} MB."
        ),
        key=_profile_picture_upload_key(),
    )

    if upload is not None and upload.size > _MAX_PROFILE_PICTURE_SIZE_BYTES:
        st.error(f"Choose a file of {_MAX_PROFILE_PICTURE_SIZE_MB} MB or smaller.")
        _clear_profile_picture_uploader()
        upload = None

    if upload is not None and st.session_state.get(_PROFILE_PENDING_DIALOG_KEY) is None:
        upload_sig = (upload.name, upload.size)
        if st.session_state.get(_PROFILE_PENDING_UPLOAD_SIG_KEY) != upload_sig:
            st.session_state[_PROFILE_PENDING_UPLOAD_SIG_KEY] = upload_sig
            _queue_profile_picture_dialog({"kind": "upload"})

    st.markdown("**Choose a preset**")
    provider_picture = st.user.to_dict().get("picture")
    show_provider = profile_pictures.is_remote(provider_picture) and not _provider_preset_is_active(
        provider_picture
    )
    preset_columns = st.columns(3 if show_provider else 2)
    with preset_columns[0]:
        _show_logo_profile_preset("black")
    with preset_columns[1]:
        _show_logo_profile_preset("white")
    if show_provider:
        with preset_columns[2]:
            _show_provider_profile_preset(provider_picture)

    pending = st.session_state.get(_PROFILE_PENDING_DIALOG_KEY)
    if isinstance(pending, dict):
        dialog_upload = upload if pending.get("kind") == "upload" else None
        _profile_picture_dialog(dialog_upload)


st.title("Personal Settings")


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
    added = user.link_account(db_client, account)
    if added:
        log_user_action(
            "Platform account linked",
            user,
            platform=platform.value,
            username=username,
            account_id=account.account_id,
        )
        st.success(f"{platform.value}-account added.")
    else:
        st.info("This account is already linked to your user.")
    st.session_state["show_add_platform_account"] = False
    st.rerun()


def _format_latest_ingestion(value: datetime | None) -> str:
    if value is None:
        return "Never"
    return value.strftime("%d %b %Y, %H:%M UTC")


def _format_pipeline_timestamp(value: datetime) -> str:
    return value.strftime("%d %b %Y, %H:%M UTC")


def _format_pipeline_result(result: str) -> str:
    return result.replace("_", " ").title()


_TIMEZONE_OPTIONS = sorted(available_timezones())


_CRON_TIME_OPTIONS = dispatch_cron_time_options()


def _show_schedule_tab() -> None:
    latest_run = user.get_latest_pipeline_run(db_client)
    st.markdown("**Latest pipeline run**")
    if latest_run is None:
        if user.latest_pipeline_run is None:
            st.caption("No pipeline run recorded yet.")
        else:
            st.caption("Latest run could not be loaded (stale reference).")
    else:
        st.write(f"**Result:** {_format_pipeline_result(latest_run.result.value)}")
        st.write(f"**Finished:** {_format_pipeline_timestamp(latest_run.finished_at)}")
        st.write(f"**Started:** {_format_pipeline_timestamp(latest_run.started_at)}")

    st.divider()

    timezone_index = (
        _TIMEZONE_OPTIONS.index(user.timezone)
        if user.timezone in _TIMEZONE_OPTIONS
        else _TIMEZONE_OPTIONS.index("UTC")
    )

    with st.form("schedule_form"):
        st.caption("Daily automatic ingestion schedule.")
        cron_time = st.selectbox(
            "Daily run time",
            options=_CRON_TIME_OPTIONS,
            index=cron_time_option_index(user.cron_time),
            format_func=format_cron_time_label,
            help=(
                "Automatic ingestion runs once per day in a 30-minute window starting "
                "at this time (your timezone below), aligned with the dispatcher schedule."
            ),
        )
        timezone = st.selectbox(
            "Timezone",
            options=_TIMEZONE_OPTIONS,
            index=timezone_index,
            help="Applies to the daily run time above.",
        )
        if st.form_submit_button("Save"):
            try:
                assert_valid_timezone(timezone)
            except ValueError as e:
                st.error(str(e))
                return

            updated_user = user
            if cron_time != user.cron_time:
                updated_user = updated_user.update_cron_time(db_client, cron_time)
            if timezone != user.timezone:
                updated_user = updated_user.update_timezone(db_client, timezone)
            set_current_user(updated_user)
            log_user_action(
                "Schedule updated",
                updated_user,
                cron_time=str(cron_time),
                timezone=timezone,
            )
            st.success("Schedule saved.")
            st.rerun()


def _show_account_list(accounts_list: list[Account]) -> None:
    column_widths = [1.5, 3, 4, 2]
    header_cols = st.columns(column_widths)
    header_cols[0].markdown("**Platform**")
    header_cols[1].markdown("**Username**")
    header_cols[2].markdown("**Latest ingestion**")
    header_cols[3].markdown("**Unlink**")

    for account in accounts_list:
        cols = st.columns(column_widths)
        with cols[0]:
            render_platform_logo(account.platform, width=24)
        cols[1].write(account.username)
        cols[2].write(_format_latest_ingestion(account.latest_ingestion))
        if cols[3].button("Unlink", key=f"unlink_{account.account_id}"):
            user.unlink_account(db_client, account)
            log_user_action(
                "Platform account unlinked",
                user,
                platform=account.platform.value,
                username=account.username,
                account_id=account.account_id,
            )
            st.success("Account unlinked.")
            st.rerun()


@st.dialog("Are you sure?")
def _safe_remove_user():
    st.warning("Your user information will be lost forever")
    if st.button("I'm sure"):
        log_user_action("User account deletion confirmed", user)
        user.unlink_all_accounts(db_client)
        user.delete_from_db(db_client)
        force_logout()


profile_tab, schedule_tab, accounts_tab, appearance_tab, danger_tab = st.tabs([
    "Profile",
    "Schedule",
    "Platform accounts",
    "Appearance",
    "Delete account",
])

with profile_tab:
    _show_profile_tab()

with schedule_tab:
    _show_schedule_tab()

with accounts_tab:
    accounts = user.get_linked_accounts(db_client)
    if accounts:
        _show_account_list(accounts)
    else:
        st.info("There are no platform accounts linked.")

    if st.button("Add new platform account"):
        st.session_state["show_add_platform_account"] = True

    if st.session_state.get("show_add_platform_account", False):
        _show_add_account_form()

with appearance_tab:
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
            st.info(
                "Streamlit is in **light** mode — the app uses your light theme until you switch."
            )
        else:
            st.info(
                "Streamlit is in **dark** mode — the app uses your dark theme until you switch."
            )

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
            log_user_action(
                "Appearance theme updated",
                user,
                light_theme_id=selected_light_id,
                dark_theme_id=selected_dark_id,
            )
            st.success("Appearance saved.")
            st.rerun()

with danger_tab:
    st.warning("Permanently delete your user and unlink all platform accounts.")
    if st.button("Remove user"):
        _safe_remove_user()
