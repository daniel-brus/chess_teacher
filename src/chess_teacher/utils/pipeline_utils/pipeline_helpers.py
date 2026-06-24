from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from chess_teacher.utils.table_data_class import TableDataClass


class PipelineResult(StrEnum):
    SKIPPED = "skipped"
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
class AggregatedPipelineRunResult:
    run_ids: tuple[str, ...]
    user_id: str | None
    account_id: str | None
    result: PipelineResult
    started_at: datetime | None
    finished_at: datetime | None
    step_results: tuple[StepResult, ...] = field(default_factory=tuple)
    run_results: tuple[PipelineRunResult, ...] = field(default_factory=tuple)

    @property
    def error_messages(self) -> tuple[str, ...]:
        return tuple(
            f"{step_result.name}: {step_result.error_message}"
            for step_result in self.step_results
            if step_result.error_message
        )

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at is None or self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()

    @property
    def latest_successful_run_id(self) -> str | None:
        successful = [
            result
            for result in self.run_results
            if result.result in {PipelineResult.SUCCESS, PipelineResult.PARTIAL}
        ]
        if not successful:
            return None
        return max(successful, key=lambda result: result.finished_at).run_id


def _uniform_optional_str(values: tuple[str | None, ...]) -> str | None:
    unique = set(values)
    if len(unique) == 1:
        return values[0]
    return None


def _aggregate_pipeline_result(results: tuple[PipelineRunResult, ...]) -> PipelineResult:
    if not results:
        return PipelineResult.SKIPPED

    statuses = {result.result for result in results}
    if PipelineResult.FAILURE in statuses:
        return PipelineResult.FAILURE
    if PipelineResult.PARTIAL in statuses:
        return PipelineResult.PARTIAL
    if statuses == {PipelineResult.SUCCESS}:
        return PipelineResult.SUCCESS
    return PipelineResult.SKIPPED


def aggregate_pipeline_run_results(
    results: list[PipelineRunResult] | tuple[PipelineRunResult, ...],
) -> AggregatedPipelineRunResult:
    """Roll up multiple pipeline run results into one summary."""
    run_results = tuple(results)
    if not run_results:
        return AggregatedPipelineRunResult(
            run_ids=(),
            user_id=None,
            account_id=None,
            result=PipelineResult.SKIPPED,
            started_at=None,
            finished_at=None,
            step_results=(),
            run_results=(),
        )

    return AggregatedPipelineRunResult(
        run_ids=tuple(result.run_id for result in run_results),
        user_id=_uniform_optional_str(tuple(result.user_id for result in run_results)),
        account_id=_uniform_optional_str(tuple(result.account_id for result in run_results)),
        result=_aggregate_pipeline_result(run_results),
        started_at=min(result.started_at for result in run_results),
        finished_at=max(result.finished_at for result in run_results),
        step_results=tuple(
            step_result for result in run_results for step_result in result.step_results
        ),
        run_results=run_results,
    )


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
