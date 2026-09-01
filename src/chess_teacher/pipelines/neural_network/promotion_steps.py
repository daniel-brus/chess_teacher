"""Pipeline steps for baseline promotion."""

from __future__ import annotations

import json

from chess_teacher.pipelines.neural_network.models import BaselineModel, BaselineModelStatus
from chess_teacher.pipelines.neural_network.promotion import (
    PromotionStrategies,
    PromotionVerdict,
)
from chess_teacher.pipelines.preprocessing.games import Game
from chess_teacher.pipelines.preprocessing.moves import Move, MoveCharacteristics
from chess_teacher.utils.db.client import DatabaseClient
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.pipeline_utils.pipeline_base import PipelineContext, PipelineStep

logger = get_logger()


def _should_skip_promotion(context: PipelineContext) -> bool:
    return bool(context.extras.get("promotion_skip"))


class LoadPromotionModelsStep(PipelineStep):
    """Load latest candidate + production baseline rows."""

    def __init__(self) -> None:
        super().__init__(name="LoadPromotionModels")

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        db_client.ensure_metadata(BaselineModel.get_metadata())
        candidate = BaselineModel.latest_with_status(db_client, BaselineModelStatus.CANDIDATE)
        production = BaselineModel.latest_with_status(db_client, BaselineModelStatus.PRODUCTION)
        context.extras["candidate_model"] = candidate
        context.extras["production_model"] = production
        if candidate is None:
            logger.info("No candidate baseline; promotion skip.")
            context.extras["promotion_skip"] = True
            return
        context.extras["promotion_skip"] = False
        logger.info(
            "Promotion models candidate=%s production=%s",
            candidate.version,
            production.version if production else None,
        )


class SampleEvalSetStep(PipelineStep):
    """Build eval datums via ``EvalSetProvider`` (random shortcut by default)."""

    def __init__(self, strategies: PromotionStrategies | None = None) -> None:
        super().__init__(name="SampleEvalSet")
        self._strategies = strategies or PromotionStrategies()

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        if _should_skip_promotion(context):
            return
        db_client.ensure_tables(
            Move.get_metadata(),
            Game.get_metadata(),
            MoveCharacteristics.get_metadata(),
        )
        production: BaselineModel | None = context.extras.get("production_model")
        # First production, or legacy (MSE/policy) production vs candidate_style: auto-promote.
        if production is None or not production.looks_like_candidate_style():
            if production is not None and not production.looks_like_candidate_style():
                logger.info(
                    "Production version=%s not candidate_style; "
                    "skip eval sample (auto-promote candidate_style candidate).",
                    production.version,
                )
                context.extras["legacy_mse_production"] = True
            context.extras["eval_datums"] = []
            logger.info("Skipping eval sample (auto-promote path).")
            return
        datums = self._strategies.eval_set.sample(db_client)
        context.extras["eval_datums"] = datums
        if not datums:
            logger.warning("Empty eval set; promotion skip.")
            context.extras["promotion_skip"] = True


class ScoreModelsStep(PipelineStep):
    """Score candidate (and production if present) via ``ModelScorer``."""

    def __init__(self, strategies: PromotionStrategies | None = None) -> None:
        super().__init__(name="ScoreModels")
        self._strategies = strategies or PromotionStrategies()

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        if _should_skip_promotion(context):
            return

        candidate: BaselineModel = context.extras["candidate_model"]
        production: BaselineModel | None = context.extras.get("production_model")
        datums = context.extras.get("eval_datums") or []
        legacy = bool(context.extras.get("legacy_mse_production"))

        if production is None or legacy:
            context.extras["candidate_score"] = None
            context.extras["production_score"] = None
            logger.info("Skipping scores; no candidate_style production baseline.")
            return

        if candidate.model_uri is None:
            raise ValueError(f"Candidate {candidate.version} has no model_uri")
        if production.model_uri is None:
            raise ValueError(f"Production {production.version} has no model_uri")

        scorer = self._strategies.scorer
        candidate_score = scorer.score(model_uri=candidate.model_uri, datums=datums)
        production_score = scorer.score(model_uri=production.model_uri, datums=datums)
        context.extras["candidate_score"] = candidate_score
        context.extras["production_score"] = production_score
        logger.info(
            "Scores candidate_primary=%s production_primary=%s details_c=%s details_p=%s",
            candidate_score.primary,
            production_score.primary,
            candidate_score.details,
            production_score.details,
        )


class DecidePromotionStep(PipelineStep):
    """Apply ``PromotionPolicy``; store verdict in context."""

    def __init__(self, strategies: PromotionStrategies | None = None) -> None:
        super().__init__(name="DecidePromotion")
        self._strategies = strategies or PromotionStrategies()

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        if _should_skip_promotion(context):
            return

        candidate = context.extras.get("candidate_model")
        production = context.extras.get("production_model")
        legacy = bool(context.extras.get("legacy_mse_production"))
        # Keep production row for archive, but decide as if no candidate_style production yet.
        has_production = production is not None and not legacy
        verdict: PromotionVerdict = self._strategies.policy.decide(
            candidate_score=context.extras.get("candidate_score"),
            production_score=context.extras.get("production_score"),
            has_candidate=candidate is not None,
            has_production=has_production,
        )
        if legacy and verdict.should_promote:
            verdict = PromotionVerdict(
                should_promote=True,
                reason=(
                    f"Legacy/non-candidate_style production="
                    f"{production.version if production else None}; "
                    "auto-promote candidate_style candidate."
                ),
                candidate_score=verdict.candidate_score,
                production_score=None,
            )
        context.extras["promotion_verdict"] = verdict
        logger.info(
            "Promotion verdict should_promote=%s reason=%s",
            verdict.should_promote,
            verdict.reason,
        )


class ApplyPromotionStep(PipelineStep):
    """Archive production and flip candidate to production when verdict says so."""

    def __init__(self) -> None:
        super().__init__(name="ApplyPromotion")

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        if _should_skip_promotion(context):
            return

        verdict: PromotionVerdict | None = context.extras.get("promotion_verdict")
        if verdict is None or not verdict.should_promote:
            logger.info("No promotion applied.")
            return

        candidate: BaselineModel = context.extras["candidate_model"]
        production: BaselineModel | None = context.extras.get("production_model")
        eval_blob = None
        if verdict.candidate_score is not None:
            eval_blob = json.dumps({
                "primary": verdict.candidate_score.primary,
                "higher_is_better": verdict.candidate_score.higher_is_better,
                **verdict.candidate_score.details,
            })
        promoted = candidate.promote_over(
            db_client,
            current_production=production,
            eval_metrics=eval_blob,
        )
        context.extras["promoted_model"] = promoted
