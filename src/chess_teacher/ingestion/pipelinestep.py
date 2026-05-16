import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from chess_teacher.ingestion.adapter import AdapterFactory
from chess_teacher.platform.account import Account
from chess_teacher.utils.db_client import DatabaseClient
from chess_teacher.utils.env_utils import get_env_variable
from chess_teacher.utils.exception_utils import (
    AdapterError,
    ConfigError,
    DatabaseError,
    FileWriteError,
)
from chess_teacher.utils.general_utils import build_daily_path, get_current_datetime
from chess_teacher.utils.pipeline_steps import PipelineStep


def _get_target_base_path(account: Account) -> Path:
    try:
        raw_dir = get_env_variable("RAW_DIR")
        if not raw_dir:
            raise ConfigError("RAW_DIR environment variable is not set")
        result = build_daily_path(Path(raw_dir) / "ingested" / account.account_id)
    except ValueError as e:
        raise ConfigError(f"RAW_DIR environment variable is not set: {e}")
    return result


class IngestionFromAPIStreamStep(PipelineStep):
    """Ingest data from an API stream into a storage location."""

    def __init__(self, account: Account):
        self.account = account
        self.target_base_path = _get_target_base_path(self.account)
        self.adapter = AdapterFactory.from_account(self.account)
        name = f"APIStreamIngestion_{self.account.account_id}"
        super().__init__(name=name)

    def _generate_filename(self) -> str:
        """Generate the name of the output file."""
        return f"{self.account.platform.value}_{uuid4().hex}.jsonl"

    def _get_last_updated(self, db_client: DatabaseClient) -> datetime | None:
        """Fetch the last updated time from the database to get the most up-to-date value."""
        try:
            result = self.account.fetch_from_db(
                db_client, id=self.account.account_id
            ).latest_ingestion
        except Exception as e:
            self.logger.log_and_raise(DatabaseError(f"Error fetching last updated time: {e}"))
        return result

    def _set_last_updated(
        self, db_client: DatabaseClient, last_updated: datetime = get_current_datetime()
    ) -> None:
        """Set the last updated time in the database."""
        try:
            self.account.upsert_latest(db_client, "latest_ingestion", last_updated)
        except Exception as e:
            self.logger.log_and_raise(DatabaseError(f"Error setting last updated time: {e}"))

    def _write_records_to_file(self, records: list[dict], path: Path) -> None:
        tmp_path = path.with_suffix(".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")
            tmp_path.rename(path)
            self.logger.info(f"[{self.name}] Written to {path}.")
        except Exception as e:
            tmp_path.unlink(missing_ok=True)
            self.logger.log_and_raise(FileWriteError(f"Error writing to file: {e}"))

    def run(self, db_client: DatabaseClient) -> None:
        output_path = self.target_base_path / self._generate_filename()
        since = self._get_last_updated(db_client)
        since_new = get_current_datetime()

        try:
            records = self.adapter.get_records(since=since)
            if not records:
                self.logger.info(f"[{self.name}] No records to write.")
                return
        except Exception as e:
            self.logger.log_and_raise(AdapterError(f"Error getting records: {e}"))

        self._write_records_to_file(records, output_path)
        self._set_last_updated(db_client, since_new)
        self.logger.info(f"[{self.name}] Ingestion completed.")
        return
