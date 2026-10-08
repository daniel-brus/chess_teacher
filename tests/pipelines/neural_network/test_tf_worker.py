"""Fit and score run inside the child. These tests stub TensorFlow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chess_teacher.pipelines.neural_network.eval_metrics import EvalMetrics
from chess_teacher.pipelines.neural_network.tf_worker import fit_model, score_model, score_weights


def test_fit_model_saves_a_file_and_returns_metrics(monkeypatch: Any) -> None:
    class _Trainer:
        def fit(
            self,
            datums: list[object],
            *,
            weights_path: Path | None,
        ) -> tuple[object, dict[str, float]]:
            assert datums == ["row"]
            assert weights_path is None
            return object(), {"n_samples": 1.0}

        @staticmethod
        def save(model: object, path: Path) -> None:
            del model
            path.write_bytes(b"weights")

    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.train.BaselineTrainer",
        _Trainer,
    )
    result = fit_model(datums=["row"], weights_path=None)  # type: ignore[arg-type]
    assert Path(result["model_path"]).read_bytes() == b"weights"
    assert result["metrics"]["n_samples"] == 1.0


def test_score_model_loads_the_candidate_and_the_parent(monkeypatch: Any) -> None:
    loads: list[Path] = []

    def load_candidate_style_keras(path: Path, *, compile_model: bool = False) -> Path:
        del compile_model
        loads.append(path)
        return path

    def score_models_on_datums(
        models: dict[str, object],
        datums: list[object],
        **_kwargs: object,
    ) -> dict[str, EvalMetrics]:
        assert datums == ["val"]
        out = {
            "candidate": EvalMetrics(
                top1_overall=0.4,
                top3_overall=0.4,
                top1_sf_agree=0.5,
                top3_sf_agree=0.5,
                top1_sf_disagree=0.2,
                top3_sf_disagree=0.2,
                top1_overall_weighted=0.4,
                n_eval=1,
                n_dropped=0,
                n_sf_agree=1,
                n_sf_disagree=0,
                sf_disagree_frac=0.0,
            )
        }
        if "parent" in models:
            out["parent"] = out["candidate"]
        return out

    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.train.load_candidate_style_keras",
        load_candidate_style_keras,
    )
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.eval_metrics.score_models_on_datums",
        score_models_on_datums,
    )
    scored = score_model(
        model_path=Path("/tmp/new.keras"),
        parent_weights_path=Path("/tmp/parent.keras"),
        datums=["val"],  # type: ignore[arg-type]
    )
    assert loads == [Path("/tmp/new.keras"), Path("/tmp/parent.keras")]
    assert scored["candidate_eval"] is not None
    assert scored["candidate_eval"].top1_overall == 0.4
    assert scored["parent_eval"] is not None


def test_score_weights_loads_one_file(monkeypatch: Any) -> None:
    loads: list[Path] = []

    def load_candidate_style_keras(path: Path, *, compile_model: bool = False) -> Path:
        del compile_model
        loads.append(path)
        return path

    def evaluate_datums(model: object, datums: list[object], **_kwargs: object) -> EvalMetrics:
        assert model == Path("/tmp/baseline.keras")
        assert datums == ["val"]
        return EvalMetrics(
            top1_overall=0.3,
            top3_overall=0.3,
            top1_sf_agree=0.4,
            top3_sf_agree=0.4,
            top1_sf_disagree=0.1,
            top3_sf_disagree=0.1,
            top1_overall_weighted=0.3,
            n_eval=1,
            n_dropped=0,
            n_sf_agree=1,
            n_sf_disagree=0,
            sf_disagree_frac=0.0,
        )

    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.train.load_candidate_style_keras",
        load_candidate_style_keras,
    )
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.eval_metrics.evaluate_datums",
        evaluate_datums,
    )
    scored = score_weights(weights_path=Path("/tmp/baseline.keras"), datums=["val"])  # type: ignore[arg-type]
    assert loads == [Path("/tmp/baseline.keras")]
    assert scored is not None
    assert scored.top1_overall == 0.3
