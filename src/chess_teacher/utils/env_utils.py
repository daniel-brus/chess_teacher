import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ENV_FILE = _PROJECT_ROOT / ".env"

# Load project .env regardless of process working directory (e.g. Streamlit).
load_dotenv(_ENV_FILE)


def get_env_variable(key: str, default: str | None = None) -> str:
    """Get an environment variable or raise an error if it's missing."""
    value = os.getenv(key, default)
    if value is None or (default is None and value.strip() == ""):
        raise ValueError(f"Missing required environment variable: {key}")
    return value


def get_hostname() -> str | None:
    """Return the HOSTNAME environment variable, or ``None`` if it is not set."""
    return os.getenv("HOSTNAME")


def get_app_port() -> str:
    """Host port for local Streamlit and Compose (``APP_PORT`` in ``.env``)."""
    return get_env_variable("APP_PORT")
