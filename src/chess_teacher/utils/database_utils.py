from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from chess_teacher.utils.env_utils import get_env_variable
from chess_teacher.utils.exception_utils import ConfigError, DatabaseError
from chess_teacher.utils.logging_utils import get_logger

logger = get_logger()


def get_db_engine(
    *,
    host: str = "",
    port: int = "",
    database: str = "",
    username: str = "",
    password: str = "",
    echo: bool = False,
) -> Engine:
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
        return create_engine(
            connection_string,
            echo=echo,
            pool_pre_ping=True,
        )
    except Exception as e:
        logger.log_and_raise(DatabaseError(f"Error occurred while creating database engine: {e}"))
