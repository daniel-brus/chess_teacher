from __future__ import annotations

import importlib
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy import inspect as sa_inspect

from src.utils.env_utils import get_env_variable
from src.utils.logging_utils import get_logger
from src.utils.metadata_utils import TableMetadata


class Pipeline:
    def __init__(self, *, metadata_path: str | Path | None = None):
        """Base class for all pipelines.
        Args:
            metadata_path: Optional explicit path to a `metadata.yml` file.
        """
        database_url = get_env_variable("DATABASE_URL")
        if not database_url:
            raise ValueError(
                "DATABASE_URL is not set. Provide it via env/.env so SQLAlchemy can connect."
            )
        self.engine = create_engine(database_url)
        self.logger = get_logger(self.__class__.__name__)
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.metadata = self._get_metadata_from_yml()
        self.schema_name = self.metadata.schema_name
        self.table_name = self.metadata.table_name
        self._ensure_tables()
        self._validate_table_schema()

    def run(self) -> None:
        """Public method to run the pipeline."""
        # TODO: Add logging, pre/post run hooks, error handling, storing pipeline run data, etc.
        self._run()

    def _run(self) -> None:
        """Run the pipeline."""
        raise NotImplementedError("Subclasses must implement the run method")

    def _ensure_tables(self) -> None:
        try:
            ddl_statements = self.metadata.ddl()
            with self.engine.begin() as conn:
                try:
                    for stmt in ddl_statements:
                        conn.execute(text(stmt))
                except Exception as e:
                    self.logger.error("Error ensuring tables with DDL: %s", ddl_statements)
                    raise e
        except Exception as e:
            self.logger.error("Unknown error ensuring tables: %s", e)

    def _get_metadata_from_yml(self) -> TableMetadata:
        """Fetch the metadata.yml file from the current directory."""
        if self.metadata_path is not None:
            path = self.metadata_path
        else:
            module = importlib.import_module(self.__class__.__module__)
            module_file = getattr(module, "__file__", None)
            if not module_file:
                raise ValueError(
                    "Cannot resolve pipeline module file. Pass metadata_path explicitly."
                )
            path = Path(module_file).resolve().parent / "metadata.yml"

        if not path.is_file():
            raise FileNotFoundError(f"metadata.yml not found at: {path}")

        self.logger.info("Loading metadata from %s", path)
        return TableMetadata.from_yaml(path)

    def _validate_table_schema(self) -> None:
        """Validate the target table schema against the metadata.yml schema."""
        try:
            insp = sa_inspect(self.engine)
            actual_cols = insp.get_columns(self.table_name, schema=self.schema_name)
            if not actual_cols:
                raise ValueError(
                    f"Table not found after creation: {self.schema_name}.{self.table_name}"
                )
        except Exception as e:
            self.logger.error(
                "Error loading table schema for %s.%s: %s",
                self.schema_name,
                self.table_name,
                e,
            )
            raise e

        try:
            actual_by_name = {c["name"]: c for c in actual_cols}
            expected_names = [c.name for c in self.metadata.columns]
            actual_names = [c["name"] for c in actual_cols]

            missing = [name for name in expected_names if name not in actual_by_name]
            extra = [name for name in actual_names if name not in set(expected_names)]
            if missing or extra:
                raise ValueError(
                    f"Actual table columns do not match yml:Missing={missing}, Extra={extra}"
                )
        except Exception as e:
            self.logger.error(
                "Error comparing table columns for %s.%s: %s",
                self.schema_name,
                self.table_name,
                e,
            )
            raise e

        type_mismatches: list[str] = []
        null_mismatches: list[str] = []
        for expected in self.metadata.columns:
            actual = actual_by_name[expected.name]
            actual_type = str(actual.get("type", "")).lower()
            expected_type = expected.data_type.lower()

            # Heuristic compare: require that the expected type token appears in actual type string.
            if expected_type not in actual_type:
                type_mismatches.append(
                    f"{expected.name} expected={expected.data_type} actual={actual.get("type")}"
                )

            # For nullability, we assume nullable if not explicitly stated in schema
            actual_nullable = bool(actual.get("nullable", True))
            if actual_nullable != expected.nullable:
                null_mismatches.append(
                    f"{expected.name} expected_nullable={expected.nullable} actual_nullable={actual_nullable}"
                )

        if type_mismatches or null_mismatches:
            err_msg = (
                "Table schema mismatch for "
                f"{self.schema_name}.{self.table_name}. "
                f"type={type_mismatches}, nullable={null_mismatches}"
            )
            self.logger.error(err_msg)
            raise ValueError(err_msg)
