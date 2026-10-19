"""Register a randomly initialized hybrid baseline for local Play.

Cloud agents have Postgres and TensorFlow, not production weights. The file
is ``HybridBoardTrainer``'s cold-start architecture with Keras's initial
weights. Re-running leaves an existing ``mock-random`` row in place and does
not archive another production model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import chess

from chess_teacher.pipelines.neural_network.board_encoder import HybridBoardTrainer
from chess_teacher.pipelines.neural_network.candidate_eval import MOVE_FEAT_DIM
from chess_teacher.pipelines.neural_network.live_state import compose_live_state_vector
from chess_teacher.pipelines.neural_network.models import (
    BaselineModel,
    BaselineModelStatus,
)
from chess_teacher.utils.db.client import DatabaseClient, get_db_client
from chess_teacher.utils.general_utils import get_current_datetime

WEIGHTS_PATH = Path("/var/lib/chess-teacher/mock_baseline.keras")
VERSION = "mock-random"


def ensure_weights(path: Path = WEIGHTS_PATH) -> Path:
    """Write the mock file when it is missing. Return the path either way."""
    if path.is_file():
        return path
    state = compose_live_state_vector(
        chess.Board(),
        evaluation_white_pov=0.0,
        last_opponent_move_uci=None,
    )
    trainer = HybridBoardTrainer()
    model = trainer.build(int(state.shape[0]))
    HybridBoardTrainer.save(model, path)
    return path


def _metrics(*, state_dim: int, max_candidates: int) -> str:
    return json.dumps({
        "head_candidate_style": 1.0,
        "encoder_hybrid_board": 1.0,
        "move_feat_dim": float(MOVE_FEAT_DIM),
        "max_candidates": float(max_candidates),
        "state_dim": float(state_dim),
        "n_samples": 0.0,
        "note": "randomly initialized weights for local Play",
    })


def register_mock_baseline(db_client: DatabaseClient, *, weights_path: Path) -> str:
    """Insert ``mock-random`` once. Return a short status word."""
    row_id = BaselineModel.generate_id({"version": VERSION})
    if BaselineModel.exists_in_db(db_client, row_id):
        stored = BaselineModel.fetch_from_db(db_client, id=row_id)
        uri = str(weights_path)
        if stored.model_uri != uri:
            replace(stored, model_uri=uri).save_to_db(db_client)
            return "updated-uri"
        return "present"

    state_dim = int(
        compose_live_state_vector(
            chess.Board(),
            evaluation_white_pov=0.0,
            last_opponent_move_uci=None,
        ).shape[0]
    )
    trainer = HybridBoardTrainer()
    current = BaselineModel.latest_with_status(db_client, BaselineModelStatus.PRODUCTION)
    status = BaselineModelStatus.ARCHIVED if current is not None else BaselineModelStatus.PRODUCTION
    BaselineModel(
        id=row_id,
        version=VERSION,
        trained_at=get_current_datetime(),
        model_uri=str(weights_path),
        status=status,
        eval_metrics=_metrics(state_dim=state_dim, max_candidates=trainer.max_candidates),
        git_commit_hash="mock",
    ).save_new_to_db(db_client)
    return status.value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="Write the Keras file and skip the database.",
    )
    args = parser.parse_args()
    path = ensure_weights()
    if args.weights_only:
        print(f"mock weights uri={path}")
        return 0
    db_client = get_db_client()
    outcome = register_mock_baseline(db_client, weights_path=path)
    print(f"mock baseline {outcome} version={VERSION} uri={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
