"""Nightly maintenance runs platform baseline training after log cleanup."""

from __future__ import annotations

from unittest.mock import MagicMock

from chess_teacher.maintenance.main import run_nightly_maintenance


def test_nightly_maintenance_trains_and_promotes_the_baseline(monkeypatch) -> None:
    maintenance = MagicMock(name="maintenance")
    training = MagicMock(name="training")
    monkeypatch.setattr(
        "chess_teacher.maintenance.main.run_maintenance",
        lambda: maintenance,
    )

    def train(*, promote: bool) -> MagicMock:
        assert promote is True
        return training

    monkeypatch.setattr(
        "chess_teacher.maintenance.main.run_baseline_training_pipeline",
        train,
    )

    assert run_nightly_maintenance() == (maintenance, training)
