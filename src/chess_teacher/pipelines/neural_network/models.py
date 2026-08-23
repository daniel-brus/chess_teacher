"""DB row classes for baseline models and training bookkeeping."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from chess_teacher.utils.db.client import DatabaseClient
from chess_teacher.utils.general_utils import generate_ident_is_literal, get_current_datetime
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.table_data_class import TableDataClass

logger = get_logger()


class BaselineModelStatus(StrEnum):
    CANDIDATE = "candidate"
    PRODUCTION = "production"
    ARCHIVED = "archived"


BASELINE_TRAINING_SCOPE = "baseline"


@dataclass(frozen=True)
class BaselineModel(TableDataClass):
    id: str
    version: str
    trained_at: datetime
    mlflow_run_id: str | None = None
    model_uri: str | None = None
    status: BaselineModelStatus = BaselineModelStatus.CANDIDATE
    parent_version: str | None = None
    data_cutoff_at: datetime | None = None
    eval_metrics: str | None = None
    git_commit_hash: str | None = None

    @classmethod
    def get_yaml_path(cls) -> Path:
        return Path(__file__).parent / "metadata.yml"

    @classmethod
    def get_key(cls) -> str:
        return "baseline_models"

    @classmethod
    def get_id_hash_columns(cls) -> tuple[str, ...]:
        return ("version",)

    @classmethod
    def latest_with_status(
        cls,
        db_client: DatabaseClient,
        status: BaselineModelStatus,
    ) -> BaselineModel | None:
        rows = cls.fetch_all_from_db(
            db_client,
            where=generate_ident_is_literal("status", status.value),
            order_by='"trained_at" DESC',
            limit=1,
        )
        return rows[0] if rows else None

    @classmethod
    def resolve_parent(cls, db_client: DatabaseClient) -> BaselineModel | None:
        """Latest candidate, else production (for incremental / cold-start)."""
        parent = cls.latest_with_status(db_client, BaselineModelStatus.CANDIDATE)
        if parent is not None:
            return parent
        return cls.latest_with_status(db_client, BaselineModelStatus.PRODUCTION)

    @classmethod
    def fetch_all_ordered(cls, db_client: DatabaseClient) -> list[BaselineModel]:
        """All baseline rows, newest first (any status)."""
        db_client.ensure_metadata(cls.get_metadata())
        return cls.fetch_all_from_db(db_client, order_by='"trained_at" DESC')

    def looks_like_policy(self) -> bool:
        """Legacy fixed-vocab policy head (superseded by candidate_style)."""
        from chess_teacher.pipelines.neural_network.move_encoding import (
            POLICY_VOCAB_SIZE,
        )

        if not self.eval_metrics:
            return False
        try:
            blob = json.loads(self.eval_metrics)
        except (TypeError, json.JSONDecodeError):
            return False
        if not isinstance(blob, dict):
            return False
        if blob.get("head_candidate_style") == 1.0 or blob.get("head") == "candidate_style":
            return False
        if blob.get("head_policy") == 1.0 or blob.get("head") == "policy":
            return True
        vocab = blob.get("vocab_size")
        if vocab is None:
            return False
        try:
            return int(float(vocab)) == POLICY_VOCAB_SIZE
        except (TypeError, ValueError):
            return False

    def looks_like_candidate_style(self) -> bool:
        """True when eval_metrics suggest candidate-aware style head (current feat dim)."""
        if not self.eval_metrics:
            return False
        try:
            blob = json.loads(self.eval_metrics)
        except (TypeError, json.JSONDecodeError):
            return False
        if not isinstance(blob, dict):
            return False
        from chess_teacher.pipelines.neural_network.candidate_eval import (
            MAX_CANDIDATES,
            MOVE_FEAT_DIM,
        )

        feat_dim = blob.get("move_feat_dim")
        if feat_dim is not None:
            try:
                if int(float(feat_dim)) != MOVE_FEAT_DIM:
                    return False
            except (TypeError, ValueError):
                return False
        elif blob.get("head_candidate_style") == 1.0 or blob.get("head") == "candidate_style":
            # Legacy candidate_style without feat dim → treat as incompatible with v2 feats.
            return False

        if blob.get("head_candidate_style") == 1.0 or blob.get("head") == "candidate_style":
            return True
        max_c = blob.get("max_candidates")
        if max_c is None:
            return False
        try:
            return int(float(max_c)) == MAX_CANDIDATES and feat_dim is not None
        except (TypeError, ValueError):
            return False

    @classmethod
    def next_version(cls, db_client: DatabaseClient) -> str:
        rows = cls.fetch_all_from_db(db_client, order_by='"trained_at" DESC', limit=50)
        max_n = 0
        for row in rows:
            if row.version.startswith("v") and row.version[1:].isdigit():
                max_n = max(max_n, int(row.version[1:]))
        return f"v{max_n + 1}"

    @staticmethod
    def current_git_commit() -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    def promote_over(
        self,
        db_client: DatabaseClient,
        *,
        current_production: BaselineModel | None,
        eval_metrics: str | None = None,
    ) -> BaselineModel:
        """Archive ``current_production`` (if any) and mark this row production."""
        if current_production is not None and current_production.id != self.id:
            archived = replace(current_production, status=BaselineModelStatus.ARCHIVED)
            archived.save_to_db(db_client)
            logger.info(
                "Archived baseline version=%s (was production)",
                current_production.version,
            )
        promoted = replace(
            self,
            status=BaselineModelStatus.PRODUCTION,
            eval_metrics=eval_metrics if eval_metrics is not None else self.eval_metrics,
        )
        promoted.save_to_db(db_client)
        logger.info("Promoted baseline version=%s to production", promoted.version)
        return promoted


@dataclass(frozen=True)
class TrainingState(TableDataClass):
    scope: str
    last_trained_data_cutoff: datetime | None = None
    last_min_data_check_at: datetime | None = None

    @classmethod
    def get_yaml_path(cls) -> Path:
        return Path(__file__).parent / "metadata.yml"

    @classmethod
    def get_key(cls) -> str:
        return "training_state"

    @classmethod
    def get_id_hash_columns(cls) -> tuple[str, ...]:
        return ()

    @classmethod
    def for_scope(cls, db_client: DatabaseClient, scope: str) -> TrainingState:
        db_client.ensure_metadata(cls.get_metadata())
        rows = cls.fetch_all_from_db(
            db_client,
            where=generate_ident_is_literal("scope", scope),
            limit=1,
        )
        if rows:
            return rows[0]
        return cls(scope=scope)

    @classmethod
    def for_baseline(cls, db_client: DatabaseClient) -> TrainingState:
        return cls.for_scope(db_client, BASELINE_TRAINING_SCOPE)

    def with_check_at(self, checked_at: datetime | None = None) -> TrainingState:
        return TrainingState(
            scope=self.scope,
            last_trained_data_cutoff=self.last_trained_data_cutoff,
            last_min_data_check_at=checked_at or get_current_datetime(),
        )

    def with_cutoff(self, cutoff: datetime, *, checked_at: datetime | None = None) -> TrainingState:
        return TrainingState(
            scope=self.scope,
            last_trained_data_cutoff=cutoff,
            last_min_data_check_at=checked_at or get_current_datetime(),
        )
