"""Pipeline framework: Pipeline and PipelineStep base classes."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from chess_teacher.utils.db_client import DatabaseClient, get_db_client
from chess_teacher.utils.exception_utils import DatabaseError, PipelineError
from chess_teacher.utils.general_utils import generate_ident_is_literal, get_current_datetime
from chess_teacher.utils.logging_utils import EnhancedLogger, get_logger
from chess_teacher.utils.table_data_class import TableDataClass

# ---------------------------------------------------------------------------
# Enums & result types
# ---------------------------------------------------------------------------


class PipelineResult(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    IN_PROGRESS = "in_progress"


# Sentinel: finished_at value written to DB to signal an active (locked) run.
_LOCK_EPOCH: datetime = datetime(1970, 1, 1, tzinfo=UTC)

# Default stale-lock timeout.
_DEFAULT_LOCK_TIMEOUT_HOURS: float = 1.0


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


@dataclass(frozen=True)
class PipelineContext:
    user_id: str | None = None
    account_id: str | None = None


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


# ---------------------------------------------------------------------------
# PipelineStep
# ---------------------------------------------------------------------------


_DEFAULT_NO_RETRY_ON: tuple[type[Exception], ...] = (
    ValueError,
    TypeError,
    AssertionError,
    NotImplementedError,
)


class PipelineStep(ABC):
    """
    Abstract base class for a single pipeline step.

    Subclasses must implement `run()`. Execution (including pre/post
    logic and retry handling) is managed by `execute()`, which is called
    by the Pipeline — not directly by the user.
    """

    def __init__(
        self,
        name: str,
        *,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        critical: bool = True,
        no_retry_on: tuple[type[Exception], ...] = _DEFAULT_NO_RETRY_ON,
        logger: EnhancedLogger | None = None,
    ) -> None:
        self.name = name
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.critical = critical
        self.no_retry_on = no_retry_on
        self.logger = logger or get_logger()

    @abstractmethod
    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        """Implement the core logic of this step."""

    def execute(self, db_client: DatabaseClient, context: PipelineContext) -> StepResult:
        """
        Run this step with retry handling.
        Called by Pipeline.run() — not directly.
        """
        self.logger.info(f"[{self.name}] Starting step.")
        started_at = get_current_datetime()

        attempt = 0
        last_error: Exception | None = None

        while attempt <= self.max_retries:
            if attempt > 0:
                wait = self.backoff_factor**attempt
                self.logger.warning(
                    f"[{self.name}] Retry {attempt}/{self.max_retries} "
                    f"after {wait:.1f}s (reason: {last_error})."
                )
                time.sleep(wait)

            try:
                self.run(db_client, context)
                finished_at = get_current_datetime()
                self.logger.info(
                    f"[{self.name}] Completed in "
                    f"{(finished_at - started_at).total_seconds():.2f}s."
                )
                return StepResult(
                    name=self.name,
                    result=PipelineResult.SUCCESS,
                    started_at=started_at,
                    finished_at=finished_at,
                )

            except Exception as e:
                last_error = e
                if isinstance(e, self.no_retry_on):
                    self.logger.error(f"[{self.name}] Non-retryable error: {e}. Aborting step.")
                    break
                attempt += 1

        # All attempts exhausted (or non-retryable error hit).
        finished_at = get_current_datetime()
        self.logger.error(f"[{self.name}] Failed after {attempt} attempt(s): {last_error}.")
        return StepResult(
            name=self.name,
            result=PipelineResult.FAILURE,
            started_at=started_at,
            finished_at=finished_at,
            error_message=str(last_error),
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    """
    Orchestrates an ordered sequence of PipelineSteps.

    Lock mechanism (via platform.pipeline_runs):
    - _acquire_lock: inserts a placeholder row with finished_at=EPOCH and
      result=IN_PROGRESS. If a non-stale row already exists for
      (user_id, account_id, name), raises immediately to prevent concurrent runs.
    - _save_run_result: overwrites that placeholder with the real result
      (save_new_to_db merges on run_id as PK), then appends per-step results.
    - Stale lock: a lock is considered abandoned when finished_at=EPOCH and
      started_at is older than `lock_timeout_hours`. Stale locks are silently
      overwritten.
    """

    def __init__(
        self,
        name: str,
        steps: list[PipelineStep],
        user_id: str | None = None,
        account_id: str | None = None,
        db_client: DatabaseClient | None = None,
        lock_timeout_hours: float = _DEFAULT_LOCK_TIMEOUT_HOURS,
        logger: EnhancedLogger | None = None,
    ) -> None:
        self.name = name
        self.context = PipelineContext(user_id=user_id, account_id=account_id)
        self.steps = steps
        self.db_client = db_client or get_db_client()
        self.lock_timeout_hours = lock_timeout_hours
        self.logger = logger or get_logger()
        self._run_id: str | None = None

    def run(self) -> PipelineRunResult:
        self.logger.info(
            f"[Pipeline:{self.name}] Starting for user {self.context.user_id} "
            f"account {self.context.account_id}."
        )
        started_at = get_current_datetime()
        step_results: tuple[StepResult, ...] = ()
        pipeline_result = PipelineResult.SUCCESS
        run_result: PipelineRunResult | None = None
        run_error: Exception | None = None

        try:
            self._pre_run(started_at)

            for step in self.steps:
                step_result = step.execute(self.db_client, self.context)
                step_results += (step_result,)

                if step_result.result == PipelineResult.FAILURE:
                    if step.critical:
                        self.logger.error(
                            f"[Pipeline:{self.name}] Critical step '{step.name}' "
                            f"failed. Aborting."
                        )
                        pipeline_result = PipelineResult.FAILURE
                        break
                    else:
                        self.logger.warning(
                            f"[Pipeline:{self.name}] Non-critical step '{step.name}' "
                            f"failed. Continuing."
                        )
                        pipeline_result = PipelineResult.PARTIAL
        except Exception as e:
            pipeline_result = PipelineResult.FAILURE
            run_error = e
        finally:
            if self._run_id is not None:
                finished_at = get_current_datetime()
                run_result = PipelineRunResult(
                    run_id=self._run_id,
                    name=self.name,
                    user_id=self.context.user_id,
                    account_id=self.context.account_id,
                    result=pipeline_result,
                    started_at=started_at,
                    finished_at=finished_at,
                    step_results=step_results,
                )
                try:
                    self._post_run(run_result)
                finally:
                    self._release_lock()

        if run_error is not None:
            self.logger.log_and_raise(
                run_error,
                f"[Pipeline:{self.name}] Failed to run: {run_error}",
            )
        elif not run_result:
            self.logger.log_and_raise(
                PipelineError(f"[Pipeline:{self.name}] Run finished without a run_id.")
            )
        else:
            self.logger.info(
                f"[Pipeline:{self.name}] Finished with result={pipeline_result} "
                f"in {run_result.duration_seconds:.2f}s."
            )
            return run_result
        assert False  # Should never happen (mypy check)

    # ------------------------------------------------------------------
    # Pre / post hooks
    # ------------------------------------------------------------------

    def _pre_run(self, started_at: datetime) -> None:
        """Checks and setup before any step runs."""
        self._check_db_connection()
        self._acquire_lock(started_at)

    def _post_run(self, result: PipelineRunResult) -> None:
        """Teardown and persistence — always runs, even on failure."""
        self._save_run_result(result)

    # ------------------------------------------------------------------
    # Lock
    # ------------------------------------------------------------------

    def _acquire_lock(self, started_at: datetime) -> None:
        """
        Register this run as active by inserting a placeholder row with
        finished_at=EPOCH. Raises if a non-stale lock exists for
        (user_id, account_id, name). Sets self._run_id on success.
        """
        meta = PipelineRunResult.get_metadata()
        self.db_client.ensure_table(meta)

        name_clause = generate_ident_is_literal("name", self.name)
        user_id_clause = generate_ident_is_literal("user_id", self.context.user_id)
        account_id_clause = generate_ident_is_literal("account_id", self.context.account_id)
        finished_at_clause = generate_ident_is_literal("finished_at", _LOCK_EPOCH.isoformat())
        active_lock_where = (
            f"{name_clause} AND {user_id_clause} AND {account_id_clause} AND {finished_at_clause}"
        )
        existing: list[dict] = self.db_client.read(  # type: ignore[assignment]
            meta,
            columns=["run_id", "started_at"],
            where=active_lock_where,
        )

        if existing:
            stale_cutoff = _as_utc(get_current_datetime()) - timedelta(
                hours=self.lock_timeout_hours
            )
            fresh_locks = [lock for lock in existing if _as_utc(lock["started_at"]) > stale_cutoff]

            if fresh_locks:
                locked_at = _as_utc(fresh_locks[0]["started_at"])
                self.logger.log_and_raise(
                    PipelineError(
                        f"[Pipeline:{self.name}] Already running for user "
                        f"{self.context.user_id!r} account {self.context.account_id!r} "
                        f"(started at {locked_at}). Aborting."
                    )
                )

            self.logger.warning(
                f"[Pipeline:{self.name}] Stale lock found "
                f"({len(existing)} active lock row(s)). Removing before acquiring a new lock."
            )
            self.db_client.delete_where(meta, where=active_lock_where)

        run_id = str(uuid4())
        PipelineRunResult(
            run_id=run_id,
            name=self.name,
            user_id=self.context.user_id,
            account_id=self.context.account_id,
            result=PipelineResult.IN_PROGRESS,  # placeholder — overwritten in _save_run_result
            started_at=started_at,
            finished_at=_LOCK_EPOCH,
        ).save_new_to_db(self.db_client)

        self._run_id = run_id
        self.logger.info(f"[Pipeline:{self.name}] Lock acquired (run_id={run_id}).")

    def _release_lock(self) -> bool:
        """
        Remove this run's active lock row if it still exists.

        The normal post-run path overwrites the lock row with the final run
        result. This cleanup only deletes rows that are still marked active.
        Returns True if cleanup completed without error, otherwise False.
        """
        if self._run_id is None:
            return True

        meta = PipelineRunResult.get_metadata()
        run_id_clause = generate_ident_is_literal("run_id", self._run_id)
        finished_at_clause = generate_ident_is_literal("finished_at", _LOCK_EPOCH.isoformat())
        active_lock_where = f"{run_id_clause} AND {finished_at_clause}"

        try:
            deleted = self.db_client.delete_where(meta, where=active_lock_where)
        except Exception as e:
            self.logger.error(
                f"[Pipeline:{self.name}] Failed to release active lock "
                f"(run_id={self._run_id}): {e}",
                exc_info=True,
            )
            return False

        if deleted:
            self.logger.warning(
                f"[Pipeline:{self.name}] Released dangling active lock " f"(run_id={self._run_id})."
            )
        return True

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _check_db_connection(self) -> None:
        """Verify DB is reachable before starting."""
        try:
            self.db_client.engine.connect()
            self.logger.info(f"[Pipeline:{self.name}] DB connection OK.")
        except Exception as e:
            self.logger.log_and_raise(
                DatabaseError(f"[Pipeline:{self.name}] DB unreachable before run: {e}")
            )

    def _save_run_result(self, result: PipelineRunResult) -> None:
        """
        Overwrite the placeholder lock row with the real result (merge on
        run_id as PK), then append per-step results. Always called — even
        on failure.
        """
        try:
            result.save_to_db(self.db_client)

            for step_order, step_result in enumerate(result.step_results):
                PipelineRunStepResult.from_step_result(
                    run_id=result.run_id,
                    step_order=step_order,
                    step_result=step_result,
                ).save_to_db(self.db_client)

            self.logger.info(f"[Pipeline:{self.name}] Run result saved (run_id={result.run_id}).")
        except Exception as e:
            self.logger.log_and_raise(
                DatabaseError(f"[Pipeline:{self.name}] Failed to save run result: {e}.")
            )
