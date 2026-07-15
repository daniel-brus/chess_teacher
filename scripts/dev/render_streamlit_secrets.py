"""Write ``.streamlit/secrets.toml`` from ``STREAMLIT_*`` environment variables.

Expects secrets to be injected by Doppler (``doppler run -- ...``) or a local
``.env`` loaded before this script runs. Used for local Streamlit and Compose
bind-mounts; production VPS uses ``apply.sh`` heredoc instead.
"""

from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SECRETS_PATH = _ROOT / ".streamlit" / "secrets.toml"
_GOOGLE_METADATA_URL = "https://accounts.google.com/.well-known/openid-configuration"

_REQUIRED = (
    "STREAMLIT_REDIRECT_URI",
    "STREAMLIT_COOKIE_SECRET",
    "STREAMLIT_GOOGLE_CLIENT_ID",
    "STREAMLIT_GOOGLE_CLIENT_SECRET",
)


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def render_secrets_toml() -> str:
    redirect_uri = _require("STREAMLIT_REDIRECT_URI")
    cookie_secret = _require("STREAMLIT_COOKIE_SECRET")
    client_id = _require("STREAMLIT_GOOGLE_CLIENT_ID")
    client_secret = _require("STREAMLIT_GOOGLE_CLIENT_SECRET")
    return (
        "[auth]\n"
        f'redirect_uri = "{redirect_uri}"\n'
        f'cookie_secret = "{cookie_secret}"\n'
        "\n"
        "[auth.google]\n"
        f'client_id = "{client_id}"\n'
        f'client_secret = "{client_secret}"\n'
        f'server_metadata_url = "{_GOOGLE_METADATA_URL}"\n'
    )


def main() -> int:
    content = render_secrets_toml()
    _SECRETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SECRETS_PATH.write_text(content, encoding="utf-8")
    print(f"Wrote {_SECRETS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
