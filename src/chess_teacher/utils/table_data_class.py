from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import MISSING, asdict, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from chess_teacher.utils.exception_utils import ConfigError, DatabaseError
from chess_teacher.utils.general_utils import (
    generate_hash,
    generate_ident_is_literal,
    generate_idents_are_literals,
)
from chess_teacher.utils.logging_utils import get_logger
from chess_teacher.utils.metadata_utils import TableMetadata

if TYPE_CHECKING:
    from chess_teacher.utils.db_client import DatabaseClient

logger = get_logger()


class TableDataClass(ABC):
    @classmethod
    @abstractmethod
    def get_key(cls) -> str: ...

    @classmethod
    @abstractmethod
    def get_yaml_path(cls) -> Path: ...

    @classmethod
    def get_metadata(cls) -> TableMetadata:
        return TableMetadata(key=cls.get_key(), yaml_path=cls.get_yaml_path())

    @classmethod
    def get_primary_key_columns(cls) -> tuple[str, ...]:
        return cls.get_metadata().primary_key

    @classmethod
    @abstractmethod
    def get_id_hash_columns(cls) -> tuple[str, ...]:
        """Field names (in order) hashed by generate_id into the stored primary key value."""
        ...

    @classmethod
    def get_timestamp_columns(cls) -> tuple[str, ...]:
        """Field names allowed for upsert_latest; override in subclasses when needed."""
        return ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        kwargs: dict[str, Any] = {}
        for dataclass_field in fields(cls):
            field_name = dataclass_field.name
            if field_name in data:
                kwargs[field_name] = data[field_name]
            elif dataclass_field.default is not MISSING:
                kwargs[field_name] = dataclass_field.default
            elif dataclass_field.default_factory is not MISSING:
                kwargs[field_name] = dataclass_field.default_factory()
            else:
                raise ValueError(
                    f'Missing required field "{field_name}" for {cls.__name__}.from_dict().'
                )
        return cls(**kwargs)

    @classmethod
    def generate_id(cls, source: dict[str, Any]) -> str:
        hash_cols = cls.get_id_hash_columns()
        for col in hash_cols:
            if col not in source:
                logger.log_and_raise(
                    ConfigError(f"Missing required field ({col}) for {cls.__name__}.generate_id().")
                )
        try:
            return generate_hash([str(source[col]) for col in hash_cols])
        except Exception:
            logger.log_and_raise(
                ConfigError(f"Hashed PK can not be generated from source: {source}")
            )

    @classmethod
    def _where_for_id(cls, row_id: str) -> str:
        pk_cols = cls.get_primary_key_columns()
        if len(pk_cols) != 1:
            logger.log_and_raise(
                ConfigError(
                    f"{cls.__name__}._where_for_id() requires a single primary key column, got {pk_cols}"
                )
            )
        return generate_ident_is_literal(pk_cols[0], row_id)

    @classmethod
    def fetch_from_db(
        cls,
        db_client: DatabaseClient,
        *,
        id: str | None = None,
        source: dict[str, Any] | None = None,
    ) -> Self:
        """
        Method to fetch an object from a databse that have ONE id column.
        Args:
            db_client: DatabaseClient
            id: value of the hash-id to fetch. If not provided: try to generate from source.
            source: dict that the id can be generated from. Should contain alls the columns to hash.
        Returns:
            Object as generated from the fetched DB entry.
        Raises:
            Exception if one arises, or if zero/multiple entries are found in the DB.
        """
        row_id = id or cls.generate_id(source or {})
        try:
            tablemetadata = cls.get_metadata()
            where = cls._where_for_id(row_id)
            result = db_client.read(tablemetadata, where=where)
            if len(result) != 1:
                logger.log_and_raise(
                    DatabaseError(
                        f"Could not find unique {cls.__name__} ({len(result)} results) in DB with {where}"
                    )
                )
        except Exception as e:
            logger.log_and_raise(e)
        return cls.from_dict(result[0])

    @classmethod
    def exists_in_db(cls, db_client: DatabaseClient, id: str) -> bool:
        """True if exactly one row matches id; False if zero. Raises otherwise."""
        try:
            tablemetadata = cls.get_metadata()
            return db_client.exists(tablemetadata, where=cls._where_for_id(id))
        except Exception as e:
            logger.log_and_raise(e)

    def get_where_clause(self) -> str:
        pk_cols = type(self).get_primary_key_columns()
        return generate_idents_are_literals(pk_cols, [getattr(self, col) for col in pk_cols])

    def save_new_to_db(self, db_client: DatabaseClient) -> None:
        try:
            tablemetadata = type(self).get_metadata()
            db_client.ensure_table(tablemetadata)
            result = db_client.insert([asdict(self)], tablemetadata, on_conflict="nothing")
            if result.rows_inserted == 1:
                logger.info(f"{type(self).__name__} {self.get_where_clause()} saved to database.")
        except Exception as e:
            logger.log_and_raise(e)

    def upsert_field(self, db_client: DatabaseClient, field: str, value: Any) -> None:
        try:
            tablemetadata = type(self).get_metadata()
            db_client.update_where(tablemetadata, {field: value}, where=self.get_where_clause())
        except Exception as e:
            logger.log_and_raise(e)

    def upsert_latest(
        self,
        db_client: DatabaseClient,
        field: str,
        ts: datetime | None = None,
    ) -> None:
        allowed_fields = type(self).get_timestamp_columns()
        if field not in allowed_fields:
            logger.log_and_raise(
                ConfigError(
                    f"Illegal field ({field}) for {type(self).__name__}.upsert_latest(); "
                    f"must be one of: {", ".join(allowed_fields)}."
                )
            )
        self.upsert_field(db_client, field, ts or datetime.now(UTC))

    def delete_from_db(self, db_client: DatabaseClient) -> None:
        try:
            tablemetadata = type(self).get_metadata()
            db_client.delete_where(tablemetadata, where=self.get_where_clause())
        except Exception as e:
            logger.log_and_raise(e)
