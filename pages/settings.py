import streamlit as st

from chess_teacher.platform.account import Account, AccountPlatform
from chess_teacher.platform.users_accounts import (
    add_account,
    get_accounts_for_user,
    remove_account,
    remove_all_accounts_for_user,
)
from chess_teacher.utils.db_client import get_db_client
from streamlit_utils.session_state import force_logout, get_current_user

user = get_current_user()

db_client = get_db_client()

st.title("Personal Settings")


def _platform_label(platform: AccountPlatform | str) -> str:
    return platform.value if isinstance(platform, AccountPlatform) else platform


def _show_add_account_form() -> None:
    with st.form("add_platform_account"):
        platform = st.selectbox(
            "Platform",
            options=list(AccountPlatform),
            format_func=lambda platform: platform.value,
        )
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
        cols[0].write(_platform_label(account.platform))
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

if st.button("Remove user"):
    _safe_remove_user()
