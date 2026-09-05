import re
from typing import Any

from psycopg.types.json import Jsonb
from sqlalchemy import Connection, create_engine, text
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.engine import URL, Engine

from chess_teacher.utils.env_utils import get_env_variable
from chess_teacher.utils.exception_utils import ConfigError, DatabaseError
from chess_teacher.utils.general_utils import quote_ident
from chess_teacher.utils.logging import get_logger

logger = get_logger()

_ALLOWED_SESSION_SETTINGS = frozenset({"max_parallel_workers_per_gather"})
_SESSION_SETTING_VALUE_RE = re.compile(r"^[0-9]+$")


def _adapt_copy_value(value: Any) -> Any:
    """Wrap JSON-like values so psycopg can COPY them into ``json``/``jsonb`` columns.

    The inline MERGE path serialises dict/list values with ``json.dumps``; the COPY
    staging path must do the equivalent or psycopg raises "cannot adapt type 'dict'".
    """
    if isinstance(value, (dict, list)):
        return Jsonb(value)
    return value


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

    def _apply_session_settings(self, conn: Connection, session_settings: dict[str, str]) -> None:
        for key, value in session_settings.items():
            if key not in _ALLOWED_SESSION_SETTINGS:
                self._logger.log_and_raise(
                    DatabaseError(f"Disallowed Postgres session setting: {key!r}")
                )
            if not _SESSION_SETTING_VALUE_RE.match(str(value)):
                self._logger.log_and_raise(
                    DatabaseError(f"Invalid Postgres session setting value: {value!r}")
                )
            conn.execute(text(f"SET LOCAL {key} = {value}"))

    # Voor parameterised queries — consumeer binnen context
    def execute_parameterized_query(
        self,
        query: str,
        params: list[dict] | dict,
        *,
        session_settings: dict[str, str] | None = None,
    ) -> list[dict]:
        try:
            with self.begin() as conn:
                if session_settings:
                    self._apply_session_settings(conn, session_settings)
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
                        copy.write_row(
                            tuple(_adapt_copy_value(record.get(c)) for c in col_names)
                        )
        except Exception as e:
            self._logger.log_and_raise(
                DatabaseError(f"Error copying records into {quoted_table}: {e}")
            )


_db_engine: EnrichedEngine | None = None


def build_postgres_url(
    *,
    host: str = "",
    port: str = "",
    database: str = "",
    username: str = "",
    password: str = "",
    sslmode: str = "",
) -> URL:
    """Build a SQLAlchemy Postgres URL from args, falling back to ``POSTGRES_*`` env."""
    try:
        host = host or get_env_variable("POSTGRES_HOST")
        port = port or get_env_variable("POSTGRES_PORT")
        database = database or get_env_variable("POSTGRES_DB")
        username = username or get_env_variable("POSTGRES_USER")
        password = password or get_env_variable("POSTGRES_PASSWORD")
        sslmode = sslmode or get_env_variable("POSTGRES_SSLMODE", default="")
    except Exception as e:
        logger.log_and_raise(
            ConfigError(f"Error occurred while fetching database credentials: {e}")
        )

    query = {"sslmode": sslmode} if sslmode else {}
    return URL.create(
        drivername="postgresql+psycopg",
        username=username,
        password=password,
        host=host,
        port=int(port),
        database=database,
        query=query,
    )


def postgres_url_string(
    *,
    host: str = "",
    port: str = "",
    database: str = "",
    username: str = "",
    password: str = "",
    sslmode: str = "",
    hide_password: bool = False,
) -> str:
    """Render the app Postgres URL as a string (e.g. for MLflow tracking)."""
    return build_postgres_url(
        host=host,
        port=port,
        database=database,
        username=username,
        password=password,
        sslmode=sslmode,
    ).render_as_string(hide_password=hide_password)


def get_db_engine(
    *,
    host: str = "",
    port: str = "",
    database: str = "",
    username: str = "",
    password: str = "",
    sslmode: str = "",
    echo: bool = False,
) -> EnrichedEngine:
    """Return a shared PostgreSQL engine for the process, or a fresh one when overridden."""
    if host or port or database or username or password or sslmode or echo:
        return _create_db_engine(
            host=host,
            port=port,
            database=database,
            username=username,
            password=password,
            sslmode=sslmode,
            echo=echo,
        )

    global _db_engine
    if _db_engine is not None:
        logger.debug("Reusing Postgres engine singleton.")
        return _db_engine

    _db_engine = _create_db_engine(
        host=host,
        port=port,
        database=database,
        username=username,
        password=password,
        sslmode=sslmode,
        echo=echo,
    )
    return _db_engine


def reset_db_engine_for_tests() -> None:
    """Dispose the module-level engine singleton (tests only)."""
    global _db_engine
    if _db_engine is not None:
        _db_engine.dispose()
    _db_engine = None


def _create_db_engine(
    *,
    host: str = "",
    port: str = "",
    database: str = "",
    username: str = "",
    password: str = "",
    sslmode: str = "",
    echo: bool = False,
) -> EnrichedEngine:
    """Create a new PostgreSQL SQLAlchemy engine."""
    try:
        url = build_postgres_url(
            host=host,
            port=port,
            database=database,
            username=username,
            password=password,
            sslmode=sslmode,
        )
        engine = create_engine(
            url,
            echo=echo,
            pool_pre_ping=True,
        )
        enriched = EnrichedEngine(engine.pool, engine.dialect, engine.url)
        logger.info(
            "Postgres engine created host=%s port=%s database=%s user=%s sslmode=%s",
            url.host,
            url.port,
            url.database,
            url.username,
            (url.query.get("sslmode") if url.query else None) or "default",
        )
    except Exception as e:
        logger.log_and_raise(DatabaseError(f"Error occurred while creating database engine: {e}"))
    return enriched
