from __future__ import annotations

from pathlib import Path

import pytest

from chess_teacher.pipelines.neural_network import train


@pytest.fixture(autouse=True)
def _clear_model_cache() -> None:
    train.clear_candidate_style_model_cache()
    yield
    train.clear_candidate_style_model_cache()


def test_load_candidate_style_from_uri_caches_in_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    weights = tmp_path / "policy.keras"
    weights.write_bytes(b"keras")

    load_calls = {"n": 0}
    sentinel = object()

    def _fake_load(_path: Path, **kwargs: object) -> object:
        load_calls["n"] += 1
        return sentinel

    class _Tracker:
        def require_keras_weights(self, model_uri: str) -> Path:
            assert model_uri == "s3://bucket/policy.keras"
            return weights

    monkeypatch.setattr(train, "load_candidate_style_keras", _fake_load)
    monkeypatch.setattr(train, "model_is_candidate_style_compatible", lambda *_a, **_k: True)

    first = train.load_candidate_style_from_uri("s3://bucket/policy.keras", tracker=_Tracker())
    second = train.load_candidate_style_from_uri("s3://bucket/policy.keras", tracker=_Tracker())

    assert first is second is sentinel
    assert load_calls["n"] == 1


def test_load_candidate_style_from_uri_reports_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    weights = tmp_path / "policy.keras"
    weights.write_bytes(b"keras")
    progress: list[str] = []

    class _Tracker:
        def require_keras_weights(self, model_uri: str) -> Path:
            return weights

    monkeypatch.setattr(train, "load_candidate_style_keras", lambda *_a, **_k: object())
    monkeypatch.setattr(train, "model_is_candidate_style_compatible", lambda *_a, **_k: True)

    train.load_candidate_style_from_uri(
        "s3://bucket/policy.keras",
        tracker=_Tracker(),
        on_progress=progress.append,
    )
    train.load_candidate_style_from_uri(
        "s3://bucket/policy.keras",
        tracker=_Tracker(),
        on_progress=progress.append,
    )

    assert progress[0] == "Fetching model weights from storage…"
    assert progress[1] == "Loading model into TensorFlow…"
    assert progress[2] == "Using cached TensorFlow model…"
