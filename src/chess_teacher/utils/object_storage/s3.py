from __future__ import annotations

import io
import re
from collections.abc import Iterator
from contextlib import contextmanager
from io import TextIOWrapper

import boto3  # type: ignore[import-untyped]
from boto3.s3.transfer import TransferConfig  # type: ignore[import-untyped]
from botocore.config import Config  # type: ignore[import-untyped]
from botocore.exceptions import ClientError  # type: ignore[import-untyped]

from chess_teacher.utils.exception_utils import FileError
from chess_teacher.utils.files.file_utils import TextStreamSource
from chess_teacher.utils.logging import EnhancedLogger, get_logger
from chess_teacher.utils.object_storage.base import ObjectStorage
from chess_teacher.utils.object_storage.keys import key_basename

_MULTIPART_THRESHOLD = 8 * 1024 * 1024
_DELETE_BATCH_SIZE = 1000
_DELETE_MISSING_CODES = frozenset({"404", "NoSuchKey", "NotFound"})


class S3ObjectStorage(ObjectStorage):
    """Object storage backed by an S3-compatible bucket."""

    def __init__(
        self,
        bucket: str,
        key_prefix: str,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
        *,
        logger: EnhancedLogger | None = None,
    ) -> None:
        self.bucket = bucket
        self.key_prefix = key_prefix.strip("/")
        self.logger = logger or get_logger()
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version="s3v4"),
        )
        self._transfer_config = TransferConfig(multipart_threshold=_MULTIPART_THRESHOLD)
        self.logger.info(
            "S3 object storage client initialized bucket=%s endpoint=%s key_prefix=%s",
            self.bucket,
            endpoint_url,
            self.key_prefix or "(root)",
        )

    def _full_key(self, relative: str) -> str:
        relative = relative.strip("/")
        if self.key_prefix:
            return self.resolve_key(self.key_prefix, relative) if relative else self.key_prefix
        return relative

    def _relative_key(self, full_key: str) -> str:
        prefix = f"{self.key_prefix}/" if self.key_prefix else ""
        if prefix and full_key.startswith(prefix):
            return full_key[len(prefix) :]
        if self.key_prefix and full_key == self.key_prefix:
            return ""
        return full_key

    @staticmethod
    def _is_not_found(exc: ClientError) -> bool:
        code = exc.response.get("Error", {}).get("Code", "")
        return code in {"404", "NoSuchKey", "NotFound"}

    @contextmanager
    def open_text(self, key: str, *, encoding: str = "utf-8-sig") -> Iterator[TextStreamSource]:
        full_key = self._full_key(key)
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=full_key)
        except ClientError as e:
            if self._is_not_found(e):
                self.logger.log_and_raise(FileError(f"Object does not exist: {key}"))
            self.logger.log_and_raise(FileError(f"Could not open {key}: {e}"))

        body = response["Body"]
        content_length = response.get("ContentLength")
        self.logger.debug(
            "S3 GET text opened bucket=%s key=%s bytes=%s",
            self.bucket,
            key,
            content_length,
        )
        try:
            stream = TextIOWrapper(body, encoding=encoding)
            yield TextStreamSource(stream, source_name=key)
        except OSError as e:
            self.logger.log_and_raise(FileError(f"Could not read {key}: {e}"))
        finally:
            body.close()

    def read_bytes(self, key: str) -> bytes | None:
        full_key = self._full_key(key)
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=full_key)
            data = response["Body"].read()
        except ClientError as e:
            if self._is_not_found(e):
                self.logger.debug("S3 GET miss bucket=%s key=%s", self.bucket, key)
                return None
            self.logger.log_and_raise(FileError(f"Could not read {key}: {e}"))
        self.logger.debug(
            "S3 GET bucket=%s key=%s bytes=%s",
            self.bucket,
            key,
            len(data),
        )
        return data

    def write_bytes(self, key: str, data: bytes, *, overwrite: bool = False) -> None:
        full_key = self._full_key(key)
        if not overwrite:
            try:
                self._client.head_object(Bucket=self.bucket, Key=full_key)
                self.logger.log_and_raise(FileError(f"Object already exists: {key}"))
            except ClientError as e:
                if not self._is_not_found(e):
                    self.logger.log_and_raise(FileError(f"Could not write {key}: {e}"))

        try:
            if len(data) >= _MULTIPART_THRESHOLD:
                self._client.upload_fileobj(
                    io.BytesIO(data),
                    self.bucket,
                    full_key,
                    Config=self._transfer_config,
                )
                upload_mode = "multipart"
            else:
                self._client.put_object(Bucket=self.bucket, Key=full_key, Body=data)
                upload_mode = "put_object"
        except ClientError as e:
            self.logger.log_and_raise(FileError(f"Could not write {key}: {e}"))
        self.logger.debug(
            "S3 PUT bucket=%s key=%s bytes=%s mode=%s overwrite=%s",
            self.bucket,
            key,
            len(data),
            upload_mode,
            overwrite,
        )

    def write_text_atomic(
        self,
        key: str,
        text: str,
        *,
        encoding: str = "utf-8",
        overwrite: bool = False,
    ) -> None:
        self.write_bytes(key, text.encode(encoding), overwrite=overwrite)

    def list_keys(
        self,
        prefix: str = "",
        *,
        recursive: bool = True,
        suffix: str | None = None,
        glob_pattern: str | None = None,
    ) -> list[str]:
        full_prefix = self._full_key(prefix)
        if full_prefix and not full_prefix.endswith("/"):
            full_prefix = f"{full_prefix}/"

        normalized_suffix = suffix if suffix is None or suffix.startswith(".") else f".{suffix}"

        def _matches_suffix(relative_key: str) -> bool:
            if normalized_suffix is None:
                return True
            name = key_basename(relative_key)
            dot = name.rfind(".")
            file_suffix = name[dot:] if dot != -1 else ""
            return file_suffix == normalized_suffix

        pattern: re.Pattern[str] | None = None
        if glob_pattern is not None:
            try:
                pattern = re.compile(glob_pattern)
            except re.error as e:
                self.logger.log_and_raise(
                    FileError(f"Invalid glob_pattern regex ({glob_pattern!r}): {e}")
                )

        paginator = self._client.get_paginator("list_objects_v2")
        keys: list[str] = []
        try:
            for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
                for obj in page.get("Contents", []):
                    object_key = obj["Key"]
                    relative = self._relative_key(object_key)
                    if not relative or relative.endswith("/"):
                        continue
                    if full_prefix and object_key.startswith(full_prefix):
                        suffix_part = object_key[len(full_prefix) :]
                    else:
                        suffix_part = object_key
                    if not recursive and "/" in suffix_part.rstrip("/"):
                        continue
                    if not _matches_suffix(relative):
                        continue
                    if pattern is not None and not pattern.search(relative):
                        continue
                    keys.append(relative)
        except ClientError as e:
            self.logger.log_and_raise(FileError(f"Could not list keys under {prefix}: {e}"))

        self.logger.debug(
            "S3 LIST bucket=%s prefix=%s keys=%s recursive=%s",
            self.bucket,
            prefix or "(root)",
            len(keys),
            recursive,
        )
        return sorted(keys)

    def move(self, source_key: str, dest_key: str, *, overwrite: bool = False) -> None:
        source_full = self._full_key(source_key)
        dest_full = self._full_key(dest_key)

        if not overwrite:
            try:
                self._client.head_object(Bucket=self.bucket, Key=dest_full)
                self.logger.log_and_raise(FileError(f"Object already exists: {dest_key}"))
            except ClientError as e:
                if not self._is_not_found(e):
                    self.logger.log_and_raise(
                        FileError(f"Could not move {source_key} to {dest_key}: {e}")
                    )

        try:
            self._client.copy_object(
                Bucket=self.bucket,
                Key=dest_full,
                CopySource={"Bucket": self.bucket, "Key": source_full},
            )
            self._client.delete_object(Bucket=self.bucket, Key=source_full)
        except ClientError as e:
            self.logger.log_and_raise(FileError(f"Could not move {source_key} to {dest_key}: {e}"))
        self.logger.debug(
            "S3 MOVE bucket=%s source=%s dest=%s overwrite=%s",
            self.bucket,
            source_key,
            dest_key,
            overwrite,
        )

    def delete(self, key: str, *, missing_ok: bool = True) -> None:
        full_key = self._full_key(key)
        try:
            self._client.delete_object(Bucket=self.bucket, Key=full_key)
        except ClientError as e:
            if missing_ok and self._is_not_found(e):
                self.logger.debug("S3 DELETE miss bucket=%s key=%s", self.bucket, key)
                return
            self.logger.log_and_raise(FileError(f"Could not delete {key}: {e}"))
        self.logger.debug("S3 DELETE bucket=%s key=%s", self.bucket, key)

    def _raise_delete_objects_errors(
        self,
        response: dict[str, object],
        *,
        relative_keys: list[str],
        full_keys: list[str],
        missing_ok: bool,
    ) -> None:
        errors_raw = response.get("Errors")
        errors: list[object] = errors_raw if isinstance(errors_raw, list) else []
        if not errors:
            return

        full_to_relative = dict(zip(full_keys, relative_keys, strict=True))
        failures: list[str] = []
        for err in errors:
            if not isinstance(err, dict):
                failures.append(str(err))
                continue
            code = str(err.get("Code", ""))
            full_key = str(err.get("Key", ""))
            relative = full_to_relative.get(full_key, self._relative_key(full_key))
            if missing_ok and code in _DELETE_MISSING_CODES:
                continue
            message = err.get("Message", "")
            failures.append(f"{relative} ({code}: {message})")

        if failures:
            self.logger.log_and_raise(FileError(f"Could not delete objects: {'; '.join(failures)}"))

    def delete_keys(self, keys: list[str], *, missing_ok: bool = True) -> None:
        if not keys:
            return

        deleted_total = 0
        for start in range(0, len(keys), _DELETE_BATCH_SIZE):
            chunk = keys[start : start + _DELETE_BATCH_SIZE]
            full_keys = [self._full_key(key) for key in chunk]
            try:
                response = self._client.delete_objects(
                    Bucket=self.bucket,
                    Delete={"Objects": [{"Key": full_key} for full_key in full_keys]},
                )
            except ClientError as e:
                if missing_ok:
                    continue
                self.logger.log_and_raise(FileError(f"Could not delete objects: {e}"))

            self._raise_delete_objects_errors(
                response,
                relative_keys=chunk,
                full_keys=full_keys,
                missing_ok=missing_ok,
            )
            deleted_raw = response.get("Deleted")
            deleted_count = len(deleted_raw) if isinstance(deleted_raw, list) else len(chunk)
            deleted_total += deleted_count

        self.logger.info(
            "S3 DELETE batch bucket=%s requested=%s deleted=%s",
            self.bucket,
            len(keys),
            deleted_total,
        )

    def presigned_get_url(self, key: str, *, expires_in: int = 3600) -> str | None:
        full_key = self._full_key(key)
        try:
            return self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": full_key},
                ExpiresIn=expires_in,
            )
        except ClientError as e:
            self.logger.log_and_raise(FileError(f"Could not presign {key}: {e}"))
