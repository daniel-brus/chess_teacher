from chess_teacher.maintenance.pipeline_steps import ClearOrphanedPipelineRunLocksStep
from chess_teacher.pipelines.pipeline_base import Pipeline


def run_maintenance() -> None:
    """Main entry point for maintenance pipeline"""
    Pipeline(
        name="maintenance",
        steps=[
            ClearOrphanedPipelineRunLocksStep(),
        ],
    ).run()
