from __future__ import annotations

import time

from chess_teacher.pipelines.neural_network.main import run_baseline_promotion_pipeline
from chess_teacher.pipelines.neural_network.tf_runtime import ensure_tensorflow_logging
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.pipeline_utils.pipeline_helpers import PipelineResult
from chess_teacher.utils.process_utils import log_script_runtime_context

# Before TF import inside the pipeline (env + Python logger wiring).
ensure_tensorflow_logging()

logger = get_logger()


def main() -> int:
    log_script_runtime_context(logger, script="baseline_promotion")
    logger.info("Baseline promotion job started.")
    started_at = time.monotonic()
    result = run_baseline_promotion_pipeline()
    duration_s = time.monotonic() - started_at

    logger.info(
        "Baseline promotion finished result=%s duration_s=%.1f run_id=%s steps=%s",
        result.result.value,
        duration_s,
        result.run_id,
        len(result.step_results),
    )
    for step_result in result.step_results:
        logger.info(
            "Promotion step summary name=%s result=%s duration_s=%.1f",
            step_result.name,
            step_result.result.value,
            step_result.duration_seconds,
        )
        if step_result.error_message:
            logger.warning(
                "Promotion step error name=%s message=%s",
                step_result.name,
                step_result.error_message,
            )

    if result.result in {PipelineResult.FAILURE, PipelineResult.PARTIAL}:
        for error_message in result.error_messages:
            logger.warning("Promotion pipeline error: %s", error_message)
        return 1
    return 0


if __name__ == "__main__":
    from chess_teacher.utils.process_utils import run_script_main

    run_script_main(main)
