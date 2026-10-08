from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from typing import Any

from chess_teacher.utils.db.client import WriteResult, WriteStrategy
from chess_teacher.utils.exception_utils import PipelineError
from chess_teacher.utils.metadata_utils import TableMetadata
from chess_teacher.utils.pipeline_utils.pipeline_base import (
    Pipeline,
    PipelineContext,
    PipelineStep,
    register_pipeline_cleanup,
)


class FakeEngine:
    def connect(self) -> None:
        return None


@dataclass
class FakeDatabaseClient:
    rows: list[dict[str, Any]] = field(default_factory=list)
    lock: Lock = field(default_factory=Lock)
    engine: FakeEngine = field(default_factory=FakeEngine)

    def ensure_table(self, table: TableMetadata) -> None:
        return None

    def ensure_metadata(self, table: TableMetadata) -> None:
        return None

    def ensure_tables(self, *tables: TableMetadata) -> None:
        return None

    def read(
        self,
        table: TableMetadata,
        *,
        columns: list[str] | None = None,
        where: str | None = None,
        **_: Any,
    ) -> list[dict[str, Any]]:
        with self.lock:
            rows = [row for row in self.rows if self._matches_where(row, where)]
            if columns is None:
                return [row.copy() for row in rows]
            return [{column: row[column] for column in columns} for row in rows]

    def insert(
        self,
        data: list[dict[str, Any]],
        table: TableMetadata,
        *,
        on_conflict: str = "error",
    ) -> WriteResult:
        rows_inserted = 0
        primary_key = tuple(table.primary_key)
        with self.lock:
            for row in data:
                row_key = tuple(row[column] for column in primary_key)
                exists = any(
                    tuple(existing[column] for column in primary_key) == row_key
                    for existing in self.rows
                )
                if exists and on_conflict == "nothing":
                    continue
                if exists:
                    raise AssertionError(f"Unexpected duplicate row for key {row_key}.")
                self.rows.append(row.copy())
                rows_inserted += 1

        return WriteResult(strategy=WriteStrategy.INSERT_IGNORE, rows_inserted=rows_inserted)

    def delete_where(self, table: TableMetadata, where: str) -> int:
        with self.lock:
            before = len(self.rows)
            self.rows = [row for row in self.rows if not self._matches_where(row, where)]
            return before - len(self.rows)

    def _matches_where(self, row: dict[str, Any], where: str | None) -> bool:
        if where is None:
            return True

        for clause in where.split(" AND "):
            if " IS NULL" in clause:
                column = clause.split(" IS NULL", maxsplit=1)[0].strip().strip('"')
                if row[column] is not None:
                    return False
                continue

            column, expected = clause.split(" = ", maxsplit=1)
            column = column.strip().strip('"')
            expected = expected.strip().strip("'")
            value = row[column]
            if hasattr(value, "isoformat"):
                value = value.isoformat()
            if str(value) != expected:
                return False

        return True


class BlockingStep(PipelineStep):
    def __init__(self, started: Event, release: Event) -> None:
        super().__init__("blocking_step")
        self.started = started
        self.release = release

    def run(self, db_client: FakeDatabaseClient, context: PipelineContext) -> None:
        self.started.set()
        self.release.wait(timeout=5)


class NoopPostRunPipeline(Pipeline):
    def _post_run(self, result) -> None:  # type: ignore[no-untyped-def]
        return None


class _FailingStep(PipelineStep):
    def __init__(self) -> None:
        super().__init__("failing", max_retries=0)

    def run(self, db_client: FakeDatabaseClient, context: PipelineContext) -> None:
        raise ValueError("boom")


class _RecordingStep(PipelineStep):
    def __init__(self, name: str, *, run_if_earlier_step_failed: bool = False) -> None:
        super().__init__(
            name,
            max_retries=0,
            run_if_earlier_step_failed=run_if_earlier_step_failed,
        )
        self.ran = False

    def run(self, db_client: FakeDatabaseClient, context: PipelineContext) -> None:
        self.ran = True


class _CleanupThenFailStep(PipelineStep):
    def __init__(self, log: list[str]) -> None:
        super().__init__("cleanup_source", max_retries=0)
        self.log = log

    def run(self, db_client: FakeDatabaseClient, context: PipelineContext) -> None:
        register_pipeline_cleanup(context, lambda: self.log.append("released"))
        raise ValueError("boom")


class _IdleProcess:
    """Records start and close without launching an interpreter."""

    def __init__(self) -> None:
        self.running = False
        self.starts = 0
        self.closes = 0

    def start(self) -> None:
        self.starts += 1
        self.running = True

    def close(self) -> None:
        self.closes += 1
        self.running = False


class _ProcessStep(PipelineStep):
    def __init__(self, name: str, process: _IdleProcess) -> None:
        super().__init__(name, max_retries=0, process=process)  # type: ignore[arg-type]

    def run(self, db_client: FakeDatabaseClient, context: PipelineContext) -> None:
        del db_client, context


class _WatchProcessStep(PipelineStep):
    def __init__(self, process: _IdleProcess) -> None:
        super().__init__("watch", max_retries=0)
        self._process = process
        self.saw_running = False

    def run(self, db_client: FakeDatabaseClient, context: PipelineContext) -> None:
        del db_client, context
        self.saw_running = self._process.running


def test_shared_process_starts_once_and_exits_after_its_last_step() -> None:
    process = _IdleProcess()
    watch = _WatchProcessStep(process)
    pipeline = NoopPostRunPipeline(
        "shared_process",
        [
            _ProcessStep("fit", process),
            watch,
            _ProcessStep("score", process),
        ],
        user_id="u1",
        account_id="a1",
        db_client=FakeDatabaseClient(),  # type: ignore[arg-type]
    )
    result = pipeline.run()
    assert result.result.value == "success"
    assert process.starts == 1
    assert watch.saw_running is True
    assert process.running is False


class _FailWithProcess(PipelineStep):
    def __init__(self, process: _IdleProcess) -> None:
        super().__init__("fit", max_retries=0, process=process)  # type: ignore[arg-type]

    def run(self, db_client: FakeDatabaseClient, context: PipelineContext) -> None:
        del db_client, context
        raise ValueError("boom")


def test_failed_holder_exits_the_process_before_later_holders() -> None:
    process = _IdleProcess()
    later = _ProcessStep("score", process)
    pipeline = NoopPostRunPipeline(
        "failed_holder",
        [_FailWithProcess(process), later],
        user_id="u1",
        account_id="a1",
        db_client=FakeDatabaseClient(),  # type: ignore[arg-type]
    )
    result = pipeline.run()
    assert result.result.value == "failure"
    assert process.starts == 1
    assert process.running is False
    assert [step.name for step in result.step_results] == ["fit"]


def test_process_is_not_started_when_its_steps_are_skipped() -> None:
    process = _IdleProcess()
    pipeline = NoopPostRunPipeline(
        "skipped_process",
        [_FailingStep(), _ProcessStep("fit", process), _ProcessStep("score", process)],
        user_id="u1",
        account_id="a1",
        db_client=FakeDatabaseClient(),  # type: ignore[arg-type]
    )
    result = pipeline.run()
    assert result.result.value == "failure"
    assert process.starts == 0
    assert process.closes == 0


def test_registered_cleanup_runs_when_a_step_fails() -> None:
    log: list[str] = []
    pipeline = NoopPostRunPipeline(
        "cleanup",
        [_CleanupThenFailStep(log)],
        user_id="u1",
        account_id="a1",
        db_client=FakeDatabaseClient(),  # type: ignore[arg-type]
    )
    result = pipeline.run()
    assert result.result.value == "failure"
    assert log == ["released"]


def test_step_flagged_to_run_after_failure_still_runs() -> None:
    skipped = _RecordingStep("skipped")
    assigned = _RecordingStep("assigned", run_if_earlier_step_failed=True)
    pipeline = NoopPostRunPipeline(
        "after_failure",
        [_FailingStep(), skipped, assigned],
        user_id="u1",
        account_id="a1",
        db_client=FakeDatabaseClient(),  # type: ignore[arg-type]
    )
    result = pipeline.run()
    assert result.result.value == "failure"
    assert [step.name for step in result.step_results] == ["failing", "assigned"]
    assert skipped.ran is False
    assert assigned.ran is True


class TestPipelineLock:
    def test_concurrent_pipeline_with_same_name_and_user_is_blocked_by_active_lock(
        self,
    ) -> None:
        db_client = FakeDatabaseClient()
        step_started = Event()
        release_step = Event()
        first_error: list[BaseException] = []
        second_error: list[BaseException] = []

        first_pipeline = NoopPostRunPipeline(
            "test_pipeline",
            [BlockingStep(step_started, release_step)],
            user_id=None,
            db_client=db_client,  # type: ignore[arg-type]
        )
        second_pipeline = NoopPostRunPipeline(
            "test_pipeline",
            [],
            user_id=None,
            db_client=db_client,  # type: ignore[arg-type]
        )

        def run_first_pipeline() -> None:
            try:
                first_pipeline.run()
            except BaseException as exc:
                first_error.append(exc)

        def run_second_pipeline() -> None:
            try:
                second_pipeline.run()
            except BaseException as exc:
                second_error.append(exc)

        first_thread = Thread(target=run_first_pipeline)
        first_thread.start()
        # Wait until the first pipeline has acquired the lock and entered the blocking step.
        assert step_started.wait(timeout=5)

        # Start the competing pipeline while the first one is still holding the active lock.
        second_thread = Thread(target=run_second_pipeline)
        second_thread.start()
        second_thread.join(timeout=5)

        try:
            # The competing pipeline should fail fast instead of running behind the first one.
            assert not second_thread.is_alive()
            assert len(second_error) == 1
            assert isinstance(second_error[0], PipelineError)
            assert "Already running" in str(second_error[0])
        finally:
            # Always release the first pipeline so a failed assertion cannot leave a thread hanging.
            release_step.set()
            first_thread.join(timeout=5)

        # The first pipeline should finish cleanly, and release its active-lock row.
        assert not first_thread.is_alive()
        assert not first_error
        assert len(db_client.rows) == 0
