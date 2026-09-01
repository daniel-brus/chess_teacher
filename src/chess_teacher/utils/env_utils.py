import os
import shutil
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


def get_environment() -> str | None:
    """Return the ENVIRONMENT variable, or ``None`` if it is not set."""
    return os.getenv("ENVIRONMENT")


def get_optional_env_variable(key: str, default: str = "") -> str:
    """Return env var value, or ``default`` when unset/empty (never raises)."""
    value = os.getenv(key)
    if value is None or value.strip() == "":
        return default
    return value


def get_app_port() -> str:
    """Host port for local Streamlit and Compose (``APP_PORT`` in ``.env``)."""
    return get_env_variable("APP_PORT")


def get_stockfish_path() -> str:
    """Path to the Stockfish binary (``STOCKFISH_PATH`` in env, else PATH lookup)."""
    explicit = get_optional_env_variable("STOCKFISH_PATH")
    if explicit and Path(explicit).is_file():
        return explicit
    on_path = shutil.which("stockfish")
    if on_path:
        return on_path
    return "stockfish"
