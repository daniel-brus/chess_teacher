from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from chess_teacher.platform.raw_assets import (
    asset_image_key,
    clear_storage_image_data_uri_cache,
    read_asset_image,
    storage_image_data_uri,
)
from chess_teacher.utils.exception_utils import FileError
from chess_teacher.utils.file_loader import FileLoaderFactory
from chess_teacher.utils.file_utils import FileType
from chess_teacher.utils.file_writer import FileWriterFactory
from chess_teacher.utils.object_storage.filesystem import FilesystemObjectStorage
from chess_teacher.utils.object_storage.health import check_raw_storage_health


@pytest.fixture
def storage() -> FilesystemObjectStorage:
    root = Path(tempfile.mkdtemp(prefix="chess_storage_test_"))
    yield FilesystemObjectStorage(root)
    shutil.rmtree(root, ignore_errors=True)


def test_write_and_read_jsonl(storage: FilesystemObjectStorage) -> None:
    writer = FileWriterFactory.get_writer(FileType.JSONL)
    records = [{"a": 1}, {"b": 2}]
    writer.write(records, "ingested/acct/2026/06/07/test.jsonl", storage)

    loader = FileLoaderFactory.get_loader(FileType.JSONL)
    loaded = loader.load_key(storage, "ingested/acct/2026/06/07/test.jsonl")
    assert loaded == records


def test_list_keys_with_suffix(storage: FilesystemObjectStorage) -> None:
    writer = FileWriterFactory.get_writer(FileType.JSONL)
    writer.write([{"x": 1}], "ingested/acct/a.jsonl", storage)
    writer.write([{"x": 2}], "ingested/acct/b.txt", storage)

    keys = storage.list_keys("ingested/acct", recursive=True, suffix="jsonl")
    assert keys == ["ingested/acct/a.jsonl"]


def test_move_and_delete_keys(storage: FilesystemObjectStorage) -> None:
    storage.write_bytes("ingested/one.jsonl", b"{}\n")
    storage.write_bytes("ingested/two.jsonl", b"{}\n")

    storage.move("ingested/one.jsonl", "processed/one.jsonl")
    assert storage.read_bytes("ingested/one.jsonl") is None
    assert storage.read_bytes("processed/one.jsonl") == b"{}\n"

    storage.delete_keys(["ingested/two.jsonl"])
    assert storage.read_bytes("ingested/two.jsonl") is None


def test_open_text_streams_line_by_line(storage: FilesystemObjectStorage) -> None:
    payload = "\n".join(json.dumps({"i": i}) for i in range(3)) + "\n"
    storage.write_text_atomic("data/lines.jsonl", payload)

    with storage.open_text("data/lines.jsonl") as source:
        lines = source.stream.readlines()

    assert len(lines) == 3


def test_read_asset_image(storage: FilesystemObjectStorage) -> None:
    storage.write_bytes(asset_image_key("test.svg"), b"<svg></svg>")
    assert read_asset_image("test.svg", storage=storage) == b"<svg></svg>"
    assert read_asset_image("missing.svg", storage=storage) is None


def test_presigned_get_url_filesystem_returns_none(storage: FilesystemObjectStorage) -> None:
    assert storage.presigned_get_url("assets/images/test.svg") is None


def test_storage_image_data_uri(storage: FilesystemObjectStorage) -> None:
    clear_storage_image_data_uri_cache()
    key = asset_image_key("logo.svg")
    storage.write_bytes(key, b"<svg></svg>")

    uri = storage_image_data_uri(key, storage=storage)
    assert uri is not None
    assert uri.startswith("data:image/svg+xml;base64,")

    cached = storage_image_data_uri(key, storage=storage)
    assert cached == uri


def test_storage_image_data_uri_process_cache(
    storage: FilesystemObjectStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_storage_image_data_uri_cache()
    key = asset_image_key("cached.svg")
    storage.write_bytes(key, b"<svg cached></svg>")

    from chess_teacher.utils.object_storage import factory

    monkeypatch.setattr(factory, "_raw_storage", storage)

    first = storage_image_data_uri(key)
    second = storage_image_data_uri(key)
    assert first == second
    assert first is second

    clear_storage_image_data_uri_cache()
    monkeypatch.setattr(factory, "_raw_storage", None)


def test_check_raw_storage_health(storage: FilesystemObjectStorage) -> None:
    check_raw_storage_health(storage)
    assert storage.list_keys("_healthcheck", recursive=True) == []


def test_check_raw_storage_health_detects_read_mismatch(
    storage: FilesystemObjectStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_read = storage.read_bytes

    def bad_read(key: str) -> bytes | None:
        data = original_read(key)
        return b"wrong" if data is not None else None

    monkeypatch.setattr(storage, "read_bytes", bad_read)

    with pytest.raises(FileError, match="expected"):
        check_raw_storage_health(storage)


def test_move_verified_clears_source(storage: FilesystemObjectStorage) -> None:
    storage.write_bytes("ingested/a.jsonl", b"{}\n")
    storage.move_verified("ingested/a.jsonl", "processed/a.jsonl")
    assert storage.read_bytes("ingested/a.jsonl") is None
    assert storage.read_bytes("processed/a.jsonl") == b"{}\n"
