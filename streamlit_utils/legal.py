"""Privacy policy and terms copy, plus sidebar/login links to those pages."""

from __future__ import annotations

import streamlit as st

from streamlit_utils.layout import app_page_link
from streamlit_utils.page_config import APP_NAME, APP_PUBLIC_URL, LEGAL_UPDATED_ON

PRIVACY_PAGE_TITLE = "Privacy policy"
TERMS_PAGE_TITLE = "Terms of use"
PRIVACY_PAGE_PATH = "streamlit_pages/privacy.py"
TERMS_PAGE_PATH = "streamlit_pages/terms.py"
SOURCE_REPO_URL = "https://github.com/daniel-brus/chess_teacher"

PRIVACY_POLICY_MARKDOWN = f"""
{APP_NAME} ({APP_PUBLIC_URL}) is a personal chess-teaching app. This notice
describes the data the app actually stores and uses. It is not legal advice.

## Who is responsible

The operator of {APP_NAME} at {APP_PUBLIC_URL} (source:
[{SOURCE_REPO_URL}]({SOURCE_REPO_URL})).

## What we collect

- **Google sign-in:** email, name, profile photo URL, and the provider subject
  id Google returns, plus whether Google marked the email as verified.
- **Your profile in this app:** display name, uploaded avatar (if you add one),
  appearance (chessboard theme), timezone, and daily ingestion schedule.
- **Linked chess accounts:** Chess.com and Lichess usernames you add.
- **Game data:** public games and related metadata pulled from those platforms
  so the app can show statistics and training features.
- **Technical logs:** page views, actions such as linking an account, and
  error logs used to operate and debug the app. Logs can include a user id
  and the server hostname.

We do not sell your data. There is no advertising network in the app.

## Cookies

Sign-in uses **strictly necessary** cookies (Streamlit session cookie and
Google OAuth). They keep you logged in and protect the login flow. We do not
set analytics or marketing cookies.

## How we use data

To create your account, keep you signed in, import your public games, show
statistics, let you play against teaching bots, and keep the service running.

## Where data is stored

Application data lives in Postgres. Avatars and other files go to object
storage (S3-compatible). Short-lived cache may use Redis. Production traffic
is served over HTTPS.

## Who we share data with

- **Google** — to authenticate you.
- **Chess.com and Lichess** — to fetch the public games for usernames you
  link. Their own terms and privacy policies apply to those platforms.
- **Hosting providers** we use to run the app (server, database, storage).

We do not use a third-party product-analytics pixel.

## How long we keep data

Until you delete your account in **Settings → Delete account**, or until we
remove the service. Cached copies expire on their own TTLs. Logs are retained
according to the server log-retention settings.

## Your choices

- Update display name, avatar, schedule, and linked accounts in **Settings**.
- Permanently delete your {APP_NAME} user (and unlink platform accounts) in
  **Settings → Delete account**.
- You can also revoke Google access in your Google account.

## Children

The app is not directed at children under 16.

## Changes

We may update this notice when the app changes. The date at the top of the
page is the latest revision ({LEGAL_UPDATED_ON}).

## Contact

Open an issue on [GitHub]({SOURCE_REPO_URL}/issues).
""".strip()

TERMS_OF_USE_MARKDOWN = f"""
These terms govern use of {APP_NAME} at {APP_PUBLIC_URL}. They are not legal
advice. If you do not agree, do not use the app.

## The service

{APP_NAME} is a personal project. It lets you sign in with Google, link
Chess.com and Lichess usernames, import public games, view statistics, and
play against teaching bots. Features can change or go offline without notice.

## Your account

You must sign in with your own Google account. Keep your Google account
secure. You are responsible for the usernames you link and for following
Chess.com and Lichess rules when you use those sites.

## Acceptable use

Do not try to break the app, scrape it aggressively, impersonate someone
else, or upload content you do not have the right to use (including profile
pictures). Do not use the app to violate chess-platform terms.

## Your content and game data

You keep whatever rights you already have in your display name, avatar, and
games. You give us permission to store and process that data so the app can
work (import, analyse, display, and play). Game records come from public
platform APIs.

## Third parties

Google, Chess.com, and Lichess are separate services. We are not responsible
for their availability, accuracy, or terms.

## No warranty

The app is provided **as is**. We do not promise that analysis, bots, or
imported games are complete, accurate, or uninterrupted.

## Limitation of liability

To the extent allowed by law, the operator is not liable for indirect or
consequential losses, or for data loss, including lost games or settings.

## Termination

You can delete your {APP_NAME} account in Settings. We may suspend access if
the app is abused or retired.

## Changes

We may update these terms. Continued use after an update means you accept
the new terms. Latest revision: {LEGAL_UPDATED_ON}.

## Contact

Open an issue on [GitHub]({SOURCE_REPO_URL}/issues).
""".strip()


def render_legal_links(*, width: str = "stretch") -> None:
    """Privacy / terms links (login screen, legal pages, sidebar)."""
    app_page_link(PRIVACY_PAGE_PATH, PRIVACY_PAGE_TITLE, width=width)
    app_page_link(TERMS_PAGE_PATH, TERMS_PAGE_TITLE, width=width)


def render_legal_page_nav() -> None:
    """Compact nav used at the top of the public legal pages."""
    from streamlit_utils.theme import apply_app_theme

    apply_app_theme(None)
    home, privacy, terms = st.columns(3)
    with home:
        app_page_link("streamlit_pages/home.py", "Home")
    with privacy:
        app_page_link(PRIVACY_PAGE_PATH, PRIVACY_PAGE_TITLE)
    with terms:
        app_page_link(TERMS_PAGE_PATH, TERMS_PAGE_TITLE)
