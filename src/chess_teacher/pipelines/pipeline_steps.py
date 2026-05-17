from chess_teacher.pipelines.pipeline_base import PipelineStep
from chess_teacher.utils.db_client import DatabaseClient


class LoadToDatabaseStep(PipelineStep):
    """Load data from arbitrary source into a table."""

    def run(self, db_client: DatabaseClient) -> None:
        pass


class StorageToTableStep(LoadToDatabaseStep):
    """Load data from storage into a table."""

    def run(self, db_client: DatabaseClient) -> None:
        pass


class TransformStep(StorageToTableStep):
    """Load data from a table, transform it and save it to another table."""

    def run(self, db_client: DatabaseClient) -> None:
        pass
