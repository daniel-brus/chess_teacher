from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import MISSING, Field, fields, is_dataclass
from datetime import UTC, date, datetime, time
from enum import StrEnum
from pathlib import Path
from types import UnionType
from typing import TYPE_CHECKING, Any, Self, Union, cast, get_args, get_origin, get_type_hints

from chess_teacher.utils.exception_utils import ConfigError, DatabaseError, MetadataError
from chess_teacher.utils.general_utils import (
    generate_hash,
    generate_ident_is_literal,
    generate_idents_are_literals,
)
from chess_teacher.utils.logging_utils import get_logger
from chess_teacher.utils.metadata_utils import TableMetadata

if TYPE_CHECKING:
    from chess_teacher.utils.db_client import DatabaseClient, WriteResult

logger = get_logger()


def _unwrap_optional_type(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin in (UnionType, Union):
        non_none = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(non_none) == 1:
            return _unwrap_optional_type(non_none[0])
    return annotation


def _is_optional_type(annotation: Any) -> bool:
    origin = get_origin(annotation)
    if origin in (UnionType, Union):
        return type(None) in get_args(annotation)
    return False


def _python_type_to_data_type(annotation: Any) -> str:
    annotation = _unwrap_optional_type(annotation)
    if annotation is str:
        return "text"
    if annotation is int:
        return "integer"
    if annotation is bool:
        return "boolean"
    if annotation is datetime:
        return "timestamp"
    if annotation is time:
        return "time"
    if isinstance(annotation, type) and issubclass(annotation, StrEnum):
        return "text"
    raise TypeError(f"Unsupported type for metadata data_type mapping: {annotation!r}")


def _normalize_default_value(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, date | datetime | time):
        return value.isoformat()
    return value


def _coerce_from_storage(value: Any, annotation: Any) -> Any:
    annotation = _unwrap_optional_type(annotation)
    if isinstance(annotation, type) and issubclass(annotation, StrEnum):
        if isinstance(value, annotation):
            return value
        return annotation(value)
    return value


def _dataclass_field_default(field: Field[Any]) -> Any:
    if field.default_factory is not MISSING:
        return field.default_factory()
    if field.default is not MISSING:
        return field.default
    return MISSING


def _dataclass_fields(cls: type[Any]) -> tuple[Field[Any], ...]:
    if not is_dataclass(cls):
        raise TypeError(f"{cls.__name__} must be a @dataclass to sync with metadata")
    return fields(cast(Any, cls))


def _persisted_dataclass_fields(cls: type[Any]) -> tuple[Field[Any], ...]:
    return tuple(field for field in _dataclass_fields(cls) if field.metadata.get("persist", True))


def _expected_nullable_for_field(field: Field[Any], type_hints: dict[str, Any]) -> bool:
    return _is_optional_type(type_hints[field.name])


def _expected_metadata_default_for_field(field: Field[Any]) -> Any:
    default = _dataclass_field_default(field)
    if default is MISSING:
        return None
    return _normalize_default_value(default)


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
    def get_dataclass_field_names(cls) -> set[str]:
        return {field.name for field in _persisted_dataclass_fields(cls)}

    @classmethod
    def validate_metadata_sync(cls) -> list[str]:
        """Return human-readable errors when dataclass fields and metadata.yml diverge."""
        metadata = cls.get_metadata()
        dc_names = cls.get_dataclass_field_names()
        meta_names = metadata.column_names()
        columns_by_name = metadata.columns_by_name()
        type_hints = get_type_hints(cls)
        errors: list[str] = []

        only_dc = sorted(dc_names - meta_names)
        only_meta = sorted(meta_names - dc_names)
        if only_dc:
            errors.append(f"only on {cls.__name__} dataclass: {only_dc}")
        if only_meta:
            errors.append(f"only in metadata.yml: {only_meta}")

        for field in _persisted_dataclass_fields(cls):
            column = columns_by_name[field.name]
            expected_type = _python_type_to_data_type(type_hints[field.name])
            if column.data_type != expected_type:
                errors.append(
                    f"{field.name}: type dataclass→{expected_type!r}, metadata→{column.data_type!r}"
                )

            expected_nullable = _expected_nullable_for_field(field, type_hints)
            if column.nullable != expected_nullable:
                errors.append(
                    f"{field.name}: nullable dataclass→{expected_nullable}, "
                    f"metadata→{column.nullable}"
                )

            expected_default = _expected_metadata_default_for_field(field)
            if column.default != expected_default:
                errors.append(
                    f"{field.name}: default dataclass→{expected_default!r}, "
                    f"metadata→{column.default!r}"
                )

        pk_cols = cls.get_primary_key_columns()
        if not pk_cols:
            errors.append(f"{cls.__name__} must declare primary_key in metadata.yml")
        else:
            missing_pk_dc = set(pk_cols) - dc_names
            missing_pk_meta = set(pk_cols) - meta_names
            if missing_pk_dc:
                errors.append(f"primary_key not on dataclass: {sorted(missing_pk_dc)}")
            if missing_pk_meta:
                errors.append(f"primary_key not in metadata columns: {sorted(missing_pk_meta)}")
            if metadata.primary_key != pk_cols:
                errors.append(
                    f"primary_key mismatch: metadata→{metadata.primary_key!r}, "
                    f"get_primary_key_columns()→{pk_cols!r}"
                )

        id_hash_cols = set(cls.get_id_hash_columns())
        unknown_id_hash = id_hash_cols - dc_names
        if unknown_id_hash:
            errors.append(f"get_id_hash_columns not on dataclass: {sorted(unknown_id_hash)}")

        timestamp_cols = set(cls.get_timestamp_columns())
        unknown_ts = timestamp_cols - dc_names
        if unknown_ts:
            errors.append(f"get_timestamp_columns not on dataclass: {sorted(unknown_ts)}")

        return errors

    @classmethod
    def assert_metadata_sync(cls) -> None:
        """Raise MetadataError if dataclass fields and metadata.yml are out of sync."""
        errors = cls.validate_metadata_sync()
        if errors:
            logger.log_and_raise(
                MetadataError(f"{cls.__name__} metadata sync failed:\n  " + "\n  ".join(errors))
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        kwargs: dict[str, Any] = {}
        type_hints = get_type_hints(cls)
        for dataclass_field in _dataclass_fields(cls):
            field_name = dataclass_field.name
            if field_name in data:
                kwargs[field_name] = _coerce_from_storage(
                    data[field_name],
                    type_hints[field_name],
                )
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
            raise

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
            result = cast(list[dict[str, Any]], db_client.read(tablemetadata, where=where))
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
            raise

    def get_where_clause(self) -> str:
        pk_cols = type(self).get_primary_key_columns()
        return generate_idents_are_literals(pk_cols, [getattr(self, col) for col in pk_cols])

    def _to_db_record(
        self,
        *,
        include_columns: list[str] | None = None,
        exclude_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        if include_columns is not None and exclude_columns is not None:
            logger.log_and_raise(ConfigError("Use include_columns or exclude_columns, not both."))

        persisted_fields = _persisted_dataclass_fields(type(self))
        field_names = {field.name for field in persisted_fields}

        if include_columns is None:
            selected_columns = set(field_names)
        else:
            selected_columns = set(include_columns)
            unknown_columns = selected_columns - field_names
            if unknown_columns:
                logger.log_and_raise(
                    ConfigError(
                        f"Unknown include_columns for {type(self).__name__}: "
                        f"{sorted(unknown_columns)}"
                    )
                )

        if exclude_columns is not None:
            excluded_columns = set(exclude_columns)
            unknown_columns = excluded_columns - field_names
            if unknown_columns:
                logger.log_and_raise(
                    ConfigError(
                        f"Unknown exclude_columns for {type(self).__name__}: "
                        f"{sorted(unknown_columns)}"
                    )
                )
            selected_columns -= excluded_columns

        primary_key_columns = set(type(self).get_primary_key_columns())
        selected_columns |= primary_key_columns

        return {
            field.name: getattr(self, field.name)
            for field in persisted_fields
            if field.name in selected_columns
        }

    def save_new_to_db(self, db_client: DatabaseClient) -> bool:
        """Save the object to the database. If the object already exists, return False."""
        try:
            tablemetadata = type(self).get_metadata()
            db_client.ensure_table(tablemetadata)
            data = self._to_db_record()
            result = db_client.insert([data], tablemetadata, on_conflict="nothing")
            if result.rows_inserted == 1:
                logger.info(f"{type(self).__name__} {self.get_where_clause()} saved to database.")
                return True
            else:
                logger.info(
                    f"{type(self).__name__} {self.get_where_clause()} already exists in database."
                )
        except Exception as e:
            logger.log_and_raise(e)
        return False

    def save_to_db(
        self,
        db_client: DatabaseClient,
        *,
        include_columns: list[str] | None = None,
        exclude_columns: list[str] | None = None,
    ) -> WriteResult:
        """
        Save the object to the database, inserting or updating as needed.

        Primary key columns are always included because merge needs them to match rows.
        """
        try:
            tablemetadata = type(self).get_metadata()
            db_client.ensure_table(tablemetadata)
            data = self._to_db_record(
                include_columns=include_columns,
                exclude_columns=exclude_columns,
            )
            result = db_client.merge(
                [data],
                tablemetadata,
                match_keys=list(tablemetadata.primary_key),
                when_matched="update",
                when_not_matched_by_target="insert",
                when_not_matched_by_source="ignore",
            )
            logger.info(f"{type(self).__name__} {self.get_where_clause()} saved to database.")
            return result
        except Exception as e:
            logger.log_and_raise(e)
            raise

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
