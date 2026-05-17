import os

from dotenv import load_dotenv

# Load .env file once
load_dotenv()


def get_env_variable(key: str, default: str | None = None) -> str:
    """Get an environment variable or raise an error if it's missing."""
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(f"Missing required environment variable: {key}")
    return value
