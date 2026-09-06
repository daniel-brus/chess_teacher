"""Unit tests for offline catch-up sibling (mocked store/trainer; no Keras)."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from chess_teacher.pipelines.neural_network import offline_catch_up
from chess_teacher.pipelines.neural_network.eval_metrics import EvalMetrics
from chess_teacher.pipelines.neural_network.pipeline_steps import MIN_NEW_MOVES_BASELINE
from chess_teacher.pipelines.neural_network.splits import GameSplitResult, SplitBucket, SplitCounts


def _metrics() -> EvalMetrics:
    return EvalMetrics(
        top1_overall=0.3,
        top3_overall=0.5,
        top1_sf_agree=0.4,
        top3_sf_agree=0.6,
        top1_sf_disagree=0.2,
        top3_sf_disagree=0.3,
        top1_overall_weighted=0.3,
        n_eval=10,
        n_dropped=0,
        n_sf_agree=5,
        n_sf_disagree=5,
        sf_disagree_frac=0.5,
    )


def _split(train: list[object]) -> GameSplitResult:
    return GameSplitResult(
        train=tuple(train),  # type: ignore[arg-type]
        val=(),
        test=(),
        salt="baseline-v1",
        counts=(
            SplitCounts(bucket=SplitBucket.TRAIN, n_games=1, n_moves=len(train)),
            SplitCounts(bucket=SplitBucket.VAL, n_games=0, n_moves=0),
            SplitCounts(bucket=SplitBucket.TEST, n_games=0, n_moves=0),
        ),
    )


def _val_datums() -> list[MagicMock]:
    return [MagicMock(game_id=f"v{i}") for i in range(10)]


def _patch_common(
    monkeypatch: pytest.MonkeyPatch,
    *,
    counts: list[int],
    fetch_max: datetime | None,
    train_datums: list[object] | None = None,
) -> dict[str, object]:
    store = MagicMock()
    store.count_since.side_effect = list(counts)
    store.fetch_since.return_value = (train_datums or [MagicMock(game_id="g1")], fetch_max)
    store_cls = MagicMock(return_value=store)

    trainer = MagicMock()
    trainer.fit.return_value = (MagicMock(), {})
    trainer_cls = MagicMock(return_value=trainer)
    trainer_cls.DEFAULT_EPOCHS = 20
    trainer_cls.save = staticmethod(
        lambda model, path: (
            Path(path).parent.mkdir(parents=True, exist_ok=True) or Path(path).write_text("k")
        )
    )

    registry = MagicMock()
    registry.exclude_holdout_games_sql.return_value = "NOT EXISTS (holdout)"
    registry.split_datums.return_value = _split(train_datums or [MagicMock(game_id="g1")])

    val_loader = MagicMock(return_value=_val_datums())
    eval_fn = MagicMock(return_value=_metrics())
    db = MagicMock()

    monkeypatch.setattr(offline_catch_up, "get_db_client", lambda: db)
    monkeypatch.setattr(offline_catch_up, "TrainingDataStore", store_cls)
    monkeypatch.setattr(offline_catch_up, "BaselineTrainer", trainer_cls)
    monkeypatch.setattr(offline_catch_up, "get_split_registry", lambda _db, split_version: registry)
    monkeypatch.setattr(offline_catch_up, "load_registry_val_datums", val_loader)
    monkeypatch.setattr(offline_catch_up, "evaluate_datums", eval_fn)
    monkeypatch.setattr(offline_catch_up, "MLflowTracker", MagicMock)

    return {
        "store": store,
        "trainer": trainer,
        "trainer_cls": trainer_cls,
        "val_loader": val_loader,
        "eval_fn": eval_fn,
        "registry": registry,
        "db": db,
    }


def _run(**overrides: object) -> int:
    kwargs: dict[str, object] = {
        "split_version": "baseline-v1",
        "val_limit": 100,
        "full_val": False,
        "max_rounds": 5,
        "min_new_moves": MIN_NEW_MOVES_BASELINE,
        "batch_limit": 100,
        "start_cutoff": None,
        "start_from_production_cutoff": False,
        "parent_uri": None,
        "epochs": 2,
        "style_disagree_boost": 2.0,
        "style_disagree_scale": 2.0,
        "output_dir": None,
    }
    kwargs.update(overrides)
    return offline_catch_up.run_offline_catch_up(**kwargs)  # type: ignore[arg-type]


def test_source_does_not_import_production_pipelines() -> None:
    src = inspect.getsource(offline_catch_up)
    assert "run_baseline_training_pipeline" not in src
    assert "run_baseline_promotion_pipeline" not in src
    assert "loop_until_caught_up" not in src
    assert "from chess_teacher.pipelines.neural_network.catch_up" not in src
    assert "from chess_teacher.pipelines.neural_network import catch_up" not in src


def test_already_caught_up_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _patch_common(
        monkeypatch,
        counts=[MIN_NEW_MOVES_BASELINE - 1],
        fetch_max=None,
    )
    assert _run() == 0
    ctx["trainer"].fit.assert_not_called()  # type: ignore[union-attr]
    ctx["val_loader"].assert_called_once()  # type: ignore[union-attr]
    ctx["store"].count_since.assert_called_once()  # type: ignore[union-attr]
    kwargs = ctx["store"].count_since.call_args.kwargs  # type: ignore[union-attr]
    assert kwargs["extra_where"] == "NOT EXISTS (holdout)"


def test_one_round_then_floor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    t1 = datetime(2026, 2, 1, tzinfo=UTC)
    train_d = MagicMock(game_id="train1")
    val_d = MagicMock(game_id="holdout_val")
    test_d = MagicMock(game_id="holdout_test")
    mixed = [train_d, val_d, test_d]
    frozen_val = _val_datums()

    store = MagicMock()
    store.count_since.side_effect = [2500, 400, 400]
    store.fetch_since.return_value = (mixed, t1)
    trainer = MagicMock()
    trainer.fit.return_value = (MagicMock(), {})
    trainer_cls = MagicMock(return_value=trainer)
    trainer_cls.save = staticmethod(
        lambda model, path: (
            Path(path).parent.mkdir(parents=True, exist_ok=True) or Path(path).write_text("k")
        )
    )
    registry = MagicMock()
    registry.exclude_holdout_games_sql.return_value = "NOT EXISTS (holdout)"
    registry.split_datums.return_value = GameSplitResult(
        train=(train_d,),  # type: ignore[arg-type]
        val=(val_d,),  # type: ignore[arg-type]
        test=(test_d,),  # type: ignore[arg-type]
        salt="baseline-v1",
        counts=(
            SplitCounts(bucket=SplitBucket.TRAIN, n_games=1, n_moves=1),
            SplitCounts(bucket=SplitBucket.VAL, n_games=1, n_moves=1),
            SplitCounts(bucket=SplitBucket.TEST, n_games=1, n_moves=1),
        ),
    )
    val_loader = MagicMock(return_value=frozen_val)
    eval_fn = MagicMock(return_value=_metrics())

    monkeypatch.setattr(offline_catch_up, "get_db_client", lambda: MagicMock())
    monkeypatch.setattr(offline_catch_up, "TrainingDataStore", MagicMock(return_value=store))
    monkeypatch.setattr(offline_catch_up, "BaselineTrainer", trainer_cls)
    monkeypatch.setattr(offline_catch_up, "get_split_registry", lambda _db, split_version: registry)
    monkeypatch.setattr(offline_catch_up, "load_registry_val_datums", val_loader)
    monkeypatch.setattr(offline_catch_up, "evaluate_datums", eval_fn)
    training_state = MagicMock()
    monkeypatch.setattr(offline_catch_up, "TrainingState", training_state)

    assert _run(output_dir=tmp_path, max_rounds=5) == 0
    assert trainer.fit.call_count == 1
    fit_datums = trainer.fit.call_args.args[0]
    assert list(fit_datums) == [train_d]
    assert val_d not in fit_datums
    assert test_d not in fit_datums
    val_loader.assert_called_once()
    assert val_loader.call_args.kwargs["assign_if_missing"] is False
    eval_fn.assert_called()
    assert eval_fn.call_args.args[1] is frozen_val
    fetch_kwargs = store.fetch_since.call_args.kwargs
    assert fetch_kwargs["extra_where"] == "NOT EXISTS (holdout)"
    assert fetch_kwargs["limit"] == 100
    assert registry.split_datums.call_args.kwargs["assign_if_missing"] is False
    training_state.for_baseline.assert_not_called()
    training_state.save_to_db.assert_not_called()


def test_stall_when_cutoff_and_count_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cutoff = datetime(2026, 1, 1, tzinfo=UTC)
    ctx = _patch_common(
        monkeypatch,
        counts=[5000, 5000],
        fetch_max=cutoff,
    )
    assert _run(start_cutoff=cutoff) == 2
    assert ctx["trainer"].fit.call_count == 1  # type: ignore[union-attr]


def test_max_rounds_returns_three(monkeypatch: pytest.MonkeyPatch) -> None:
    times = [
        datetime(2026, 3, 1, tzinfo=UTC),
        datetime(2026, 3, 2, tzinfo=UTC),
    ]
    store = MagicMock()
    store.count_since.return_value = 5000
    store.fetch_since.side_effect = [
        ([MagicMock(game_id="g1")], times[0]),
        ([MagicMock(game_id="g2")], times[1]),
    ]
    trainer = MagicMock()
    trainer.fit.return_value = (MagicMock(), {})
    trainer_cls = MagicMock(return_value=trainer)
    trainer_cls.save = staticmethod(
        lambda model, path: Path(path).parent.mkdir(parents=True, exist_ok=True)
    )
    registry = MagicMock()
    registry.exclude_holdout_games_sql.return_value = "NOT EXISTS (holdout)"
    registry.split_datums.side_effect = lambda datums, assign_if_missing=False: _split(list(datums))

    val_loader = MagicMock(return_value=_val_datums())
    monkeypatch.setattr(offline_catch_up, "get_db_client", lambda: MagicMock())
    monkeypatch.setattr(offline_catch_up, "TrainingDataStore", MagicMock(return_value=store))
    monkeypatch.setattr(offline_catch_up, "BaselineTrainer", trainer_cls)
    monkeypatch.setattr(offline_catch_up, "get_split_registry", lambda _db, split_version: registry)
    monkeypatch.setattr(offline_catch_up, "load_registry_val_datums", val_loader)
    monkeypatch.setattr(offline_catch_up, "evaluate_datums", lambda *_a, **_k: _metrics())

    assert _run(max_rounds=2) == 3
    assert trainer.fit.call_count == 2
    val_loader.assert_called_once()


def test_start_from_production_cutoff_is_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prod_cutoff = datetime(2026, 4, 1, tzinfo=UTC)
    state = MagicMock()
    state.last_trained_data_cutoff = prod_cutoff
    state.save_to_db = MagicMock()
    monkeypatch.setattr(
        offline_catch_up.TrainingState,
        "for_baseline",
        classmethod(lambda cls, db: state),
    )
    ctx = _patch_common(
        monkeypatch,
        counts=[MIN_NEW_MOVES_BASELINE - 1],
        fetch_max=None,
    )
    assert _run(start_from_production_cutoff=True) == 0
    state.save_to_db.assert_not_called()
    cutoff_arg = ctx["store"].count_since.call_args.args[0]  # type: ignore[union-attr]
    assert cutoff_arg == prod_cutoff


def test_mutex_start_cutoff_flags() -> None:
    parser = offline_catch_up.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--start-cutoff",
            "2026-01-01T00:00:00",
            "--start-from-production-cutoff",
        ])


def test_parser_has_no_hidden_or_promote() -> None:
    help_text = offline_catch_up.build_arg_parser().format_help()
    assert "--hidden" not in help_text
    assert "--promote" not in help_text
    assert "--val-limit" in help_text
    assert "--batch-limit" in help_text


def test_run_rejects_mutex_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(offline_catch_up, "get_db_client", lambda: MagicMock())
    assert (
        _run(
            start_cutoff=datetime(2026, 1, 1, tzinfo=UTC),
            start_from_production_cutoff=True,
        )
        == 1
    )
