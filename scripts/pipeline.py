from chess_teacher.ingestion.pipeline_steps import (
    IngestionFromAPIStreamStep,  # , ArchiveIngestedFilesStep
)
from chess_teacher.pipelines.pipeline_base import Pipeline, PipelineRunResult
from chess_teacher.platform.account import Account
from chess_teacher.platform.user import User


def run_pipeline(user: User, account: Account) -> PipelineRunResult:
    pipeline = Pipeline(
        name=f"Ingestion_{user.user_id}_{account.account_id}",
        steps=[
            IngestionFromAPIStreamStep(account),
            #   ArchiveIngestedFilesStep(source_path=get_env_variable("INGESTED_DIR"), archive_path=get_env_variable("ARCHIVED_DIR"))
        ],
        user_id=user.user_id,
    )
    return pipeline.run()
