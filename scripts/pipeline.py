from chess_teacher.ingestion.main import run_ingestion_pipeline
from chess_teacher.pipelines.pipeline_base import PipelineRunResult
from chess_teacher.platform.account import Account
from chess_teacher.platform.user import User


def run_pipeline(user: User, account: Account) -> PipelineRunResult:
    # TODO: add other pipelines here
    return run_ingestion_pipeline(user.user_id, account)
