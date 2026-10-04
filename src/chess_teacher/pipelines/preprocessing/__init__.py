"""Preprocessing pipeline: raw_games -> games + moves.

The package init stays empty so importing a model (for example ``games``) does
not load ``main``. ``main`` includes ``AssignGameSplitsStep``, and the split
registry imports those models.
"""

from __future__ import annotations

from typing import Any

__all__ = ["run_preprocessing_pipeline"]


def __getattr__(name: str) -> Any:
    if name == "run_preprocessing_pipeline":
        from chess_teacher.pipelines.preprocessing.main import run_preprocessing_pipeline

        return run_preprocessing_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
