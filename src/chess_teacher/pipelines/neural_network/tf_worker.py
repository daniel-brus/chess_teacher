"""Child interpreter for the two training steps that need TensorFlow.

Stdout is the pickle protocol. Logging and ``print`` go to stderr, which the
pipeline process already inherits.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chess_teacher.pipelines.neural_network.create_training_set import TrainingDatum
    from chess_teacher.pipelines.neural_network.eval_metrics import EvalMetrics


def fit_model(
    *,
    datums: list[TrainingDatum],
    weights_path: Path | None,
) -> dict[str, Any]:
    """Fit in this process and return the saved file plus the train metrics."""
    from chess_teacher.pipelines.neural_network.train import BaselineTrainer

    model, metrics = BaselineTrainer().fit(datums, weights_path=weights_path)
    out_path = Path(tempfile.mkdtemp(prefix="scheme_model_")) / "model.keras"
    BaselineTrainer.save(model, out_path)
    return {"model_path": str(out_path), "metrics": metrics}


def score_model(
    *,
    model_path: Path,
    parent_weights_path: Path | None,
    datums: list[TrainingDatum],
) -> dict[str, EvalMetrics | None]:
    """Score the saved candidate and, when present, the parent weights."""
    from chess_teacher.pipelines.neural_network.eval_metrics import score_models_on_datums
    from chess_teacher.pipelines.neural_network.train import load_candidate_style_keras

    models: dict[str, Any] = {
        "candidate": load_candidate_style_keras(Path(model_path), compile_model=False),
    }
    if parent_weights_path is not None:
        models["parent"] = load_candidate_style_keras(
            Path(parent_weights_path),
            compile_model=False,
        )
    scored = score_models_on_datums(models, datums)
    return {
        "candidate_eval": scored["candidate"],
        "parent_eval": scored.get("parent"),
    }


def main() -> None:
    """Serve fit and score until the parent asks the process to exit."""
    protocol_out = sys.stdout.buffer
    sys.stdout = sys.stderr
    from chess_teacher.utils.pipeline_utils.step_process import serve

    serve({"fit": fit_model, "score": score_model}, protocol_out)


if __name__ == "__main__":
    main()
