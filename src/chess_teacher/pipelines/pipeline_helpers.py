from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from chess_teacher.utils.table_data_class import TableDataClass


class PipelineResult(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    IN_PROGRESS = "in_progress"


@dataclass(frozen=True)
class StepResult:
    name: str
    result: PipelineResult
    started_at: datetime
    finished_at: datetime
    error_message: str | None = None

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()


@dataclass(frozen=True)
class PipelineRunResult(TableDataClass):
    run_id: str
    name: str
    user_id: str | None
    account_id: str | None
    result: PipelineResult
    started_at: datetime
    finished_at: datetime
    step_results: tuple[StepResult, ...] = field(
        default_factory=tuple,
        metadata={"persist": False},
    )

    @property
    def error_messages(self) -> tuple[str, ...]:
        return tuple(
            f"{sr.name}: {sr.error_message}" for sr in self.step_results if sr.error_message
        )

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()

    @classmethod
    def get_key(cls) -> str:
        return "pipeline_runs"

    @classmethod
    def get_yaml_path(cls) -> Path:
        return Path(__file__).parent / "metadata.yml"

    @classmethod
    def get_id_hash_columns(cls) -> tuple[str, ...]:
        return ()


@dataclass(frozen=True)
class PipelineRunStepResult(TableDataClass):
    run_id: str
    step_order: int
    name: str
    result: PipelineResult
    started_at: datetime
    finished_at: datetime
    error_message: str | None = None

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()

    @classmethod
    def from_step_result(
        cls,
        *,
        run_id: str,
        step_order: int,
        step_result: StepResult,
    ) -> PipelineRunStepResult:
        return cls(
            run_id=run_id,
            step_order=step_order,
            name=step_result.name,
            result=step_result.result,
            started_at=step_result.started_at,
            finished_at=step_result.finished_at,
            error_message=step_result.error_message,
        )

    @classmethod
    def get_key(cls) -> str:
        return "pipeline_run_steps"

    @classmethod
    def get_yaml_path(cls) -> Path:
        return Path(__file__).parent / "metadata.yml"

    @classmethod
    def get_id_hash_columns(cls) -> tuple[str, ...]:
        return ()


class ProgressWindow(Protocol):
    """
    Protocol for reporting progress during a pipeline run.

    Implementations can target different platforms (Streamlit, CLI, logging, etc.).
    The pipeline and steps use this interface — they never import platform-specific code.
    """

    def next(self, message: str) -> None:
        """Add a new message."""
        ...

    def update(self, message: str) -> None:
        """Overwrite the last message (e.g. progress counter updating)."""
        ...

    def pop(self, amount: int = 1) -> None:
        """Remove the last message(s) (e.g. drop a transient in-progress line)."""
        ...

    def success(self, message: str) -> None:
        """Report a successful outcome."""
        ...

    def warning(self, message: str) -> None:
        """Report a warning."""
        ...

    def error(self, message: str) -> None:
        """Report an error."""
        ...

    def clear(self) -> None:
        """Clear all messages."""
        ...
