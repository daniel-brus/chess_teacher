from typing import Any

from sqlalchemy import Connection, create_engine, text
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.engine import Engine

from chess_teacher.utils.env_utils import get_env_variable
from chess_teacher.utils.exception_utils import ConfigError, DatabaseError
from chess_teacher.utils.general_utils import quote_ident
from chess_teacher.utils.logging_utils import get_logger

logger = get_logger()


class EnrichedEngine(Engine):
    """Custom SQLAlchemy engine. Contains helper methods for common database operations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = (
            get_logger()
        )  # use underscore to avoid conflict with SQLAlchemy's logger property

    def get_inspector(self):
        """Return a SQLAlchemy inspector for the engine."""
        try:
            return sa_inspect(self)
        except Exception as e:
            self._logger.log_and_raise(DatabaseError(f"Error creating database inspector: {e}"))

    def execute_statements(self, statements: list[str]) -> None:
        """Execute a list of SQL statements in a transaction."""
        try:
            with self.begin() as conn:
                for stmt in statements:
                    conn.execute(text(stmt))
        except Exception as e:
            self._logger.log_and_raise(DatabaseError(f"Error executing SQL statements: {e}"))

    # Voor parameterised queries — consumeer binnen context
    def execute_parameterized_query(self, query: str, params: list[dict] | dict) -> list[dict]:
        try:
            with self.begin() as conn:
                result = conn.execute(text(query), params)
                # Consume result before closing connection
                return_list = [dict(r) for r in result.mappings().all()]
        except Exception as e:
            self._logger.log_and_raise(DatabaseError(f"Error executing parameterized query: {e}"))
        return return_list

    def execute_write(self, query: str, params: list[dict] | dict) -> int:
        """Execute write query, return affected row count."""
        try:
            with self.begin() as conn:
                result = conn.execute(text(query), params)
                return_list = result.rowcount if result.rowcount >= 0 else 0
        except Exception as e:
            self._logger.log_and_raise(DatabaseError(f"Error executing write query: {e}"))
        return return_list

    def copy_records(
        self,
        conn: Connection,
        table_name: str,
        col_names: list[str],
        records: list[dict],
    ) -> None:
        """Bulk-load rows into an existing table via psycopg3 COPY (same transaction as conn)."""
        if not records:
            return
        quoted_table = quote_ident(table_name)
        quoted_cols = ", ".join(quote_ident(c) for c in col_names)
        copy_sql = f"COPY {quoted_table} ({quoted_cols}) FROM STDIN"
        try:
            raw_conn: Any = conn.connection.driver_connection
            with raw_conn.cursor() as cursor:
                with cursor.copy(copy_sql) as copy:
                    for record in records:
                        copy.write_row(tuple(record.get(c) for c in col_names))
        except Exception as e:
            self._logger.log_and_raise(
                DatabaseError(f"Error copying records into {quoted_table}: {e}")
            )


def get_db_engine(
    *,
    host: str = "",
    port: str = "",
    database: str = "",
    username: str = "",
    password: str = "",
    echo: bool = False,
) -> EnrichedEngine:
    """
    Create PostgreSQL SQLAlchemy engine.
    """

    try:
        host = host or get_env_variable("POSTGRES_HOST")
        port = port or get_env_variable("POSTGRES_PORT")
        database = database or get_env_variable("POSTGRES_DB")
        username = username or get_env_variable("POSTGRES_USER")
        password = password or get_env_variable("POSTGRES_PASSWORD")

    except Exception as e:
        logger.log_and_raise(
            ConfigError(f"Error occurred while fetching database credentials: {e}")
        )

    try:
        connection_string = f"postgresql+psycopg://{username}:{password}@{host}:{port}/{database}"
        engine = create_engine(
            connection_string,
            echo=echo,
            pool_pre_ping=True,
        )
        enriched = EnrichedEngine(engine.pool, engine.dialect, engine.url)
    except Exception as e:
        logger.log_and_raise(DatabaseError(f"Error occurred while creating database engine: {e}"))
    return enriched
