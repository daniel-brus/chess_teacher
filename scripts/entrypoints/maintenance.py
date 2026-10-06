# Remove orphaned accounts (no link to existing users)

# Remove orphaned pipeline runs (finished_at EPOCH, started long enough ago)

from __future__ import annotations

import time

from chess_teacher.maintenance.main import (
    resolve_baseline_nightly_rounds,
    run_nightly_maintenance,
)
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.pipeline_utils.pipeline_helpers import (
    PipelineResult,
    PipelineRunResult,
)
from chess_teacher.utils.process_utils import log_script_runtime_context

logger = get_logger()


def main() -> int:
    log_script_runtime_context(logger, script="maintenance")
    logger.info("Maintenance job started.")
    started_at = time.monotonic()
    baseline_rounds = resolve_baseline_nightly_rounds()
    logger.info("Nightly baseline round budget=%s", baseline_rounds)
    maintenance_result, training_results = run_nightly_maintenance(
        baseline_rounds=baseline_rounds,
    )
    duration_s = time.monotonic() - started_at

    failed = False
    labeled_results: list[tuple[str, PipelineRunResult]] = [
        ("maintenance", maintenance_result),
    ]
    for index, training_result in enumerate(training_results, start=1):
        labeled_results.append((f"baseline_training round {index}", training_result))
    for label, result in labeled_results:
        logger.info(
            "%s finished result=%s duration_s=%.1f run_id=%s steps=%s",
            label,
            result.result.value,
            duration_s,
            result.run_id,
            len(result.step_results),
        )
        for step_result in result.step_results:
            logger.info(
                "%s step summary name=%s result=%s duration_s=%.1f",
                label,
                step_result.name,
                step_result.result.value,
                step_result.duration_seconds,
            )
            if step_result.error_message:
                logger.warning(
                    "%s step error name=%s message=%s",
                    label,
                    step_result.name,
                    step_result.error_message,
                )
        if result.result in {PipelineResult.FAILURE, PipelineResult.PARTIAL}:
            failed = True
            for error_message in result.error_messages:
                logger.warning("%s pipeline error: %s", label, error_message)

    return 1 if failed else 0


if __name__ == "__main__":
    from chess_teacher.utils.process_utils import run_script_main

    run_script_main(main)
