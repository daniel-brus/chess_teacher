import os


class Secrets:
    def __init__(self):
        self.POSTGRES_USER = self._get("POSTGRES_USER")
        self.POSTGRES_PASSWORD = self._get("POSTGRES_PASSWORD")
        self.POSTGRES_DB = self._get("POSTGRES_DB")
        self.POSTGRES_HOST = self._get("POSTGRES_HOST")
        self.POSTGRES_PORT = self._get("POSTGRES_PORT")

        self.DATABASE_URL = (
            f"postgresql://{self.POSTGRES_USER}:"
            f"{self.POSTGRES_PASSWORD}@"
            f"{self.POSTGRES_HOST}:"
            f"{self.POSTGRES_PORT}/"
            f"{self.POSTGRES_DB}"
        )

    def _get(self, key):
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Missing env var: {key}")
        return value


secrets = Secrets()
