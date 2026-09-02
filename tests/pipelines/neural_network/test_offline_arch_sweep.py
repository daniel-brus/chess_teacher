"""Unit tests for offline arch sweep (mocked fit; no Keras)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from chess_teacher.pipelines.neural_network import offline_arch_sweep
from chess_teacher.pipelines.neural_network.candidate_eval import (
    CANDIDATE_MOVE_FEAT_VERSION,
    MOVE_FEAT_DIM,
)
from chess_teacher.pipelines.neural_network.eval_metrics import EvalMetrics
from chess_teacher.pipelines.neural_network.splits import (
    GameSplitResult,
    SplitBucket,
    SplitCounts,
)


def _metrics(*, disagree: float) -> EvalMetrics:
    return EvalMetrics(
        top1_overall=0.25,
        top3_overall=0.4,
        top1_sf_agree=0.4,
        top3_sf_agree=0.5,
        top1_sf_disagree=disagree,
        top3_sf_disagree=0.2,
        top1_overall_weighted=0.25,
        n_eval=20,
        n_dropped=0,
        n_sf_agree=10,
        n_sf_disagree=10,
        sf_disagree_frac=0.5,
    )


def _split() -> GameSplitResult:
    train = [MagicMock(game_id=f"t{i}") for i in range(30)]
    val = [MagicMock(game_id=f"v{i}") for i in range(10)]
    return GameSplitResult(
        train=tuple(train),  # type: ignore[arg-type]
        val=tuple(val),  # type: ignore[arg-type]
        test=(),
        salt="baseline-v1",
        counts=(
            SplitCounts(bucket=SplitBucket.TRAIN, n_games=30, n_moves=30),
            SplitCounts(bucket=SplitBucket.VAL, n_games=10, n_moves=10),
            SplitCounts(bucket=SplitBucket.TEST, n_games=0, n_moves=0),
        ),
    )


def test_grid_is_four_cold_start_cells() -> None:
    assert offline_arch_sweep.ARCH_SWEEP_GRID == (
        (128, 64),
        (128, 128),
        (256, 64),
        (256, 128),
    )


def test_run_arch_sweep_cold_starts_each_cell(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    load = MagicMock(return_value=_split())
    monkeypatch.setattr(offline_arch_sweep, "load_registry_split", load)
    monkeypatch.setattr(offline_arch_sweep, "get_db_client", lambda: MagicMock())

    created: list[dict[str, object]] = []
    fit_parents: list[object] = []

    class FakeTrainer:
        DEFAULT_EPOCHS = 20

        def __init__(self, **kwargs: object) -> None:
            created.append(kwargs)

        def fit(
            self, datums: object, *, weights_path: object = None
        ) -> tuple[MagicMock, dict[str, float]]:
            fit_parents.append(weights_path)
            model = MagicMock()
            model.count_params.return_value = 12345
            return model, {}

    monkeypatch.setattr(offline_arch_sweep, "BaselineTrainer", FakeTrainer)
    monkeypatch.setattr(
        offline_arch_sweep,
        "evaluate_datums",
        lambda model, datums: _metrics(disagree=0.22),
    )

    assert (
        offline_arch_sweep.run_arch_sweep(
            limit=200,
            epochs=3,
            split_version="baseline-v1",
            style_disagree_boost=2.0,
            style_disagree_scale=2.0,
        )
        == 0
    )

    load.assert_called_once()
    assert len(created) == 4
    assert fit_parents == [None, None, None, None]
    hiddens = {(int(c["hidden"]), int(c["score_hidden"])) for c in created}
    assert hiddens == set(offline_arch_sweep.ARCH_SWEEP_GRID)
    for kwargs in created:
        assert "move_feat_dim" not in kwargs
        assert kwargs["epochs"] == 3

    out = capsys.readouterr().out
    assert f"feat_version={CANDIDATE_MOVE_FEAT_VERSION}" in out
    assert f"feat_dim={MOVE_FEAT_DIM}" in out
    assert "Feat layout frozen" in out


def test_parser_has_no_feat_version_flags() -> None:
    help_text = offline_arch_sweep.build_arg_parser().format_help()
    assert "--feat-version" not in help_text
    assert "--feat_version" not in help_text
    assert "--move-feat-dim" not in help_text
    assert "--salt" in help_text
    assert "--split-version" in help_text
