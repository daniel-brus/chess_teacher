"""Unit tests for offline catch-up sibling (mocked store/trainer; no Keras)."""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from chess_teacher.pipelines.neural_network import offline_catch_up
from chess_teacher.pipelines.neural_network.eval_metrics import EvalMetrics
from chess_teacher.pipelines.neural_network.pipeline_steps import MIN_NEW_MOVES_BASELINE


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


def _val_datums() -> list[MagicMock]:
    return [MagicMock(game_id=f"v{i}") for i in range(10)]


def _patch_common(
    monkeypatch: pytest.MonkeyPatch,
    *,
    counts: list[int],
    train_datums: list[object] | None = None,
    game_ids: list[str] | None = None,
    marked: int = 1,
) -> dict[str, object]:
    datums = train_datums or [MagicMock(game_id="g1")]
    ids = game_ids or ["g1"]
    store = MagicMock()
    store.count_unprocessed_train.side_effect = list(counts)
    store.fetch_unprocessed_train_batch.return_value = (datums, ids)
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
    registry.mark_processed.return_value = marked

    val_loader = MagicMock(return_value=_val_datums())
    packed = MagicMock(n_input=10, kept_datums=_val_datums())
    pack_fn = MagicMock(return_value=packed)
    eval_fn = MagicMock(return_value=_metrics())
    db = MagicMock()

    monkeypatch.setattr(offline_catch_up, "get_db_client", lambda: db)
    monkeypatch.setattr(offline_catch_up, "TrainingDataStore", store_cls)
    monkeypatch.setattr(offline_catch_up, "BaselineTrainer", trainer_cls)
    monkeypatch.setattr(offline_catch_up, "get_split_registry", lambda _db, split_version: registry)
    monkeypatch.setattr(offline_catch_up, "load_registry_val_datums", val_loader)
    monkeypatch.setattr(offline_catch_up, "pack_datums_for_eval", pack_fn)
    monkeypatch.setattr(offline_catch_up, "evaluate_packed", eval_fn)
    monkeypatch.setattr(offline_catch_up, "MLflowTracker", MagicMock)

    return {
        "store": store,
        "trainer": trainer,
        "trainer_cls": trainer_cls,
        "val_loader": val_loader,
        "pack_fn": pack_fn,
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
    assert "from chess_teacher.pipelines.neural_network.models import TrainingState" not in src
    assert "TrainingState.for_baseline" not in src
    assert "fetch_since" not in src
    assert "count_since" not in src
    assert "HashScanCursor" not in src
    assert "fetch_hash_scan_batch" not in src
    assert "count_hash_scan_pending" not in src
    assert "mark_processed" in src


def test_already_caught_up_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _patch_common(
        monkeypatch,
        counts=[MIN_NEW_MOVES_BASELINE - 1],
    )
    assert _run() == 0
    ctx["trainer"].fit.assert_not_called()  # type: ignore[union-attr]
    ctx["val_loader"].assert_called_once()  # type: ignore[union-attr]
    ctx["store"].count_unprocessed_train.assert_called_once()  # type: ignore[union-attr]
    kwargs = ctx["store"].count_unprocessed_train.call_args.kwargs  # type: ignore[union-attr]
    assert kwargs["split_version"] == "baseline-v1"
    ctx["store"].fetch_unprocessed_train_batch.assert_not_called()  # type: ignore[union-attr]
    ctx["registry"].mark_processed.assert_not_called()  # type: ignore[union-attr]


def test_one_round_then_floor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    train_d = MagicMock(game_id="train1")
    frozen_val = _val_datums()

    store = MagicMock()
    store.count_unprocessed_train.side_effect = [2500, 400, 400]
    store.fetch_unprocessed_train_batch.return_value = ([train_d], ["train1"])
    trainer = MagicMock()
    trainer.fit.return_value = (MagicMock(), {})
    trainer_cls = MagicMock(return_value=trainer)
    trainer_cls.save = staticmethod(
        lambda model, path: (
            Path(path).parent.mkdir(parents=True, exist_ok=True) or Path(path).write_text("k")
        )
    )
    registry = MagicMock()
    registry.mark_processed.return_value = 1
    val_loader = MagicMock(return_value=frozen_val)
    packed = MagicMock(n_input=10, kept_datums=frozen_val)
    pack_fn = MagicMock(return_value=packed)
    eval_fn = MagicMock(return_value=_metrics())

    monkeypatch.setattr(offline_catch_up, "get_db_client", lambda: MagicMock())
    monkeypatch.setattr(offline_catch_up, "TrainingDataStore", MagicMock(return_value=store))
    monkeypatch.setattr(offline_catch_up, "BaselineTrainer", trainer_cls)
    monkeypatch.setattr(offline_catch_up, "get_split_registry", lambda _db, split_version: registry)
    monkeypatch.setattr(offline_catch_up, "load_registry_val_datums", val_loader)
    monkeypatch.setattr(offline_catch_up, "pack_datums_for_eval", pack_fn)
    monkeypatch.setattr(offline_catch_up, "evaluate_packed", eval_fn)

    assert _run(output_dir=tmp_path, max_rounds=5) == 0
    assert trainer.fit.call_count == 1
    fit_datums = trainer.fit.call_args.args[0]
    assert list(fit_datums) == [train_d]
    val_loader.assert_called_once()
    assert val_loader.call_args.kwargs["full"] is False
    assert val_loader.call_args.kwargs["limit"] == 100
    pack_fn.assert_called_once_with(frozen_val)
    eval_fn.assert_called()
    assert eval_fn.call_args.args[1] is packed
    fetch_kwargs = store.fetch_unprocessed_train_batch.call_args.kwargs
    assert fetch_kwargs["split_version"] == "baseline-v1"
    assert fetch_kwargs["limit"] == 100
    registry.mark_processed.assert_called_once_with(["train1"])
    assert not (tmp_path / "hash_scan_cursor.json").exists()


def test_skip_does_not_mark(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _patch_common(monkeypatch, counts=[MIN_NEW_MOVES_BASELINE - 1])
    assert _run() == 0
    ctx["registry"].mark_processed.assert_not_called()  # type: ignore[union-attr]


def test_stall_when_mark_updates_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _patch_common(
        monkeypatch,
        counts=[5000],
        marked=0,
    )
    assert _run() == 2
    assert ctx["trainer"].fit.call_count == 1  # type: ignore[union-attr]
    ctx["registry"].mark_processed.assert_called_once()  # type: ignore[union-attr]


def test_stall_when_count_does_not_drop(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _patch_common(
        monkeypatch,
        counts=[5000, 5000],
        marked=1,
    )
    assert _run() == 2
    assert ctx["trainer"].fit.call_count == 1  # type: ignore[union-attr]
    ctx["registry"].mark_processed.assert_called_once()  # type: ignore[union-attr]


def test_max_rounds_returns_three(monkeypatch: pytest.MonkeyPatch) -> None:
    store = MagicMock()
    store.count_unprocessed_train.side_effect = [5000, 4000, 4000, 3000]
    store.fetch_unprocessed_train_batch.side_effect = [
        ([MagicMock(game_id="g1")], ["g1"]),
        ([MagicMock(game_id="g2")], ["g2"]),
    ]
    trainer = MagicMock()
    trainer.fit.return_value = (MagicMock(), {})
    trainer_cls = MagicMock(return_value=trainer)
    trainer_cls.save = staticmethod(
        lambda model, path: Path(path).parent.mkdir(parents=True, exist_ok=True)
    )
    registry = MagicMock()
    registry.mark_processed.return_value = 1

    val_loader = MagicMock(return_value=_val_datums())
    packed = MagicMock(n_input=10, kept_datums=_val_datums())
    monkeypatch.setattr(offline_catch_up, "get_db_client", lambda: MagicMock())
    monkeypatch.setattr(offline_catch_up, "TrainingDataStore", MagicMock(return_value=store))
    monkeypatch.setattr(offline_catch_up, "BaselineTrainer", trainer_cls)
    monkeypatch.setattr(offline_catch_up, "get_split_registry", lambda _db, split_version: registry)
    monkeypatch.setattr(offline_catch_up, "load_registry_val_datums", val_loader)
    monkeypatch.setattr(offline_catch_up, "pack_datums_for_eval", MagicMock(return_value=packed))
    monkeypatch.setattr(offline_catch_up, "evaluate_packed", lambda *_a, **_k: _metrics())

    assert _run(max_rounds=2) == 3
    assert trainer.fit.call_count == 2
    assert registry.mark_processed.call_count == 2
    val_loader.assert_called_once()


def test_save_failure_does_not_mark(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _patch_common(monkeypatch, counts=[5000])
    ctx["trainer_cls"].save = staticmethod(  # type: ignore[union-attr]
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full"))
    )
    with pytest.raises(OSError, match="disk full"):
        _run()
    ctx["registry"].mark_processed.assert_not_called()  # type: ignore[union-attr]
    ctx["eval_fn"].assert_not_called()  # type: ignore[union-attr]


def test_eval_failure_does_not_mark(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _patch_common(monkeypatch, counts=[5000])
    ctx["eval_fn"].side_effect = RuntimeError("eval boom")  # type: ignore[union-attr]
    with pytest.raises(RuntimeError, match="eval boom"):
        _run()
    ctx["registry"].mark_processed.assert_not_called()  # type: ignore[union-attr]


def test_mark_runs_after_save_and_eval(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    order: list[str] = []
    ctx = _patch_common(monkeypatch, counts=[2500, 400, 400])
    orig_fit = ctx["trainer"].fit  # type: ignore[union-attr]
    orig_fit.side_effect = lambda *a, **k: order.append("fit") or (MagicMock(), {})
    orig_eval = ctx["eval_fn"]  # type: ignore[union-attr]
    orig_eval.side_effect = lambda *a, **k: order.append("eval") or _metrics()
    ctx["trainer_cls"].save = staticmethod(  # type: ignore[union-attr]
        lambda model, path: (
            order.append("save")
            or Path(path).parent.mkdir(parents=True, exist_ok=True)
            or Path(path).write_text("k")
        )
    )
    ctx["registry"].mark_processed.side_effect = (  # type: ignore[union-attr]
        lambda ids: order.append("mark") or 1
    )
    assert _run(output_dir=tmp_path, max_rounds=5) == 0
    assert order == ["fit", "save", "eval", "mark"]
    curve = tmp_path / "val_curve.jsonl"
    assert curve.is_file()
    assert "round" in curve.read_text(encoding="utf-8")


def test_parser_has_no_hidden_or_promote_or_end_time_cutoff() -> None:
    help_text = offline_catch_up.build_arg_parser().format_help()
    assert "--hidden" not in help_text
    assert "--promote" not in help_text
    assert "--start-cutoff" not in help_text
    assert "--start-from-production-cutoff" not in help_text
    assert "--val-limit" in help_text
    assert "--batch-limit" in help_text
    assert "--parent-uri" in help_text
