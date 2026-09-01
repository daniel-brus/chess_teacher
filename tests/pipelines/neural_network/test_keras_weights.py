"""Tests for Keras weight URI resolution (no MLflow / TensorFlow)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from chess_teacher.pipelines.neural_network.keras_weights import (
    parse_s3_uri,
    resolve_keras_weights_path,
    s3_key_to_storage_relative,
)


def test_parse_s3_uri() -> None:
    assert parse_s3_uri("s3://bucket/mlflow/run/model.keras") == (
        "bucket",
        "mlflow/run/model.keras",
    )
    assert parse_s3_uri("file:///tmp/x.keras") is None


def test_s3_key_to_storage_relative_strips_prefix() -> None:
    assert s3_key_to_storage_relative("root/mlflow/x.keras", key_prefix="root") == "mlflow/x.keras"


def test_resolve_keras_weights_local_file(tmp_path: Path) -> None:
    weights = tmp_path / "m.keras"
    weights.write_bytes(b"keras")
    assert resolve_keras_weights_path(str(weights)) == weights
    assert resolve_keras_weights_path(f"file:{weights}") == weights


def test_resolve_keras_weights_s3_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    uri = "s3://chess-teacher/chess-teacher/mlflow/run/model.keras"
    storage = MagicMock()
    storage.read_bytes.return_value = b"keras-bytes"

    monkeypatch.setenv("S3_BUCKET", "chess-teacher")
    monkeypatch.setenv("STORAGE_ROOT", "chess-teacher")
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://localhost:9000")
    monkeypatch.setenv("S3_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("S3_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("CHESS_TEACHER_MODEL_CACHE_DIR", str(tmp_path / "cache"))

    path = resolve_keras_weights_path(uri, storage=storage)
    assert path is not None
    assert path.is_file()
    assert path.read_bytes() == b"keras-bytes"
    storage.read_bytes.assert_called_once_with("mlflow/run/model.keras")
