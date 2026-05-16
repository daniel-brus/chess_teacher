from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from typing import Any

from chess_teacher.utils.db_client import WriteResult, WriteStrategy
from chess_teacher.utils.exception_utils import PipelineError
from chess_teacher.utils.metadata_utils import TableMetadata
from chess_teacher.utils.pipeline_utils import Pipeline, PipelineStep


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

    def delete_where(self, table: TableMetadata, where: str) -> None:
        with self.lock:
            self.rows.clear()

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

    def run(self, db_client: FakeDatabaseClient) -> None:
        self.started.set()
        self.release.wait(timeout=5)


class NoopPostRunPipeline(Pipeline):
    def _post_run(self, result) -> None:  # type: ignore[no-untyped-def]
        return None


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

        # The first pipeline should finish cleanly, with only its active-lock row written.
        assert not first_thread.is_alive()
        assert not first_error
        assert len(db_client.rows) == 1
