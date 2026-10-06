"""User catch-up queue: personal flag, min-data gate, no production writes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from chess_teacher.pipelines.neural_network.eval_metrics import EvalMetrics
from chess_teacher.pipelines.neural_network.models import PROCESSED_FLAG_PERSONAL
from scripts.ops.offline_user_catch_up import run_offline_user_catch_up


def _metrics() -> EvalMetrics:
    return EvalMetrics(
        top1_overall=0.5,
        top3_overall=0.8,
        top1_sf_agree=0.6,
        top3_sf_agree=0.9,
        top1_sf_disagree=0.4,
        top3_sf_disagree=0.7,
        top1_overall_weighted=0.5,
        n_eval=12,
        n_dropped=0,
        n_sf_agree=6,
        n_sf_disagree=6,
        sf_disagree_frac=0.5,
    )


def test_missing_parent_returns_before_db(tmp_path: Path) -> None:
    with patch("scripts.ops.offline_user_catch_up.get_db_client") as db:
        code = run_offline_user_catch_up(
            account_id="acct-1",
            split_version="baseline-v1",
            parent_weights=str(tmp_path / "missing.keras"),
            val_limit=None,
            max_rounds=1,
            min_new_moves=300,
            batch_limit=100,
            epochs=1,
            style_disagree_boost=4.0,
            style_disagree_scale=2.0,
            output_dir=str(tmp_path),
        )
    assert code == 1
    db.assert_not_called()


def test_below_min_gate_skips_finetune(tmp_path: Path, capsys: object) -> None:
    parent = tmp_path / "parent.keras"
    parent.write_bytes(b"x")
    store = MagicMock()
    store.count_unprocessed_train.return_value = 40
    registry = MagicMock()

    with (
        patch("scripts.ops.offline_user_catch_up.get_db_client", return_value=MagicMock()),
        patch(
            "scripts.ops.offline_user_catch_up.load_account_registry_bucket_datums",
            return_value=[object()] * 12,
        ),
        patch("scripts.ops.offline_user_catch_up.TrainingDataStore", return_value=store),
        patch("scripts.ops.offline_user_catch_up.get_split_registry", return_value=registry),
        patch("scripts.ops.offline_user_catch_up.HybridBoardTrainer") as trainer_cls,
        patch("scripts.ops.offline_user_catch_up.load_hybrid_board_keras") as load_model,
    ):
        code = run_offline_user_catch_up(
            account_id="acct-1",
            split_version="baseline-v1",
            parent_weights=str(parent),
            val_limit=None,
            max_rounds=2,
            min_new_moves=300,
            batch_limit=100,
            epochs=1,
            style_disagree_boost=4.0,
            style_disagree_scale=2.0,
            output_dir=str(tmp_path / "out"),
        )

    assert code == 0
    trainer_cls.assert_not_called()
    load_model.assert_not_called()
    registry.mark_processed.assert_not_called()
    assert store.count_unprocessed_train.call_args.kwargs["flag_column"] == PROCESSED_FLAG_PERSONAL
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "no rounds" in out
    assert "no_training_state_write=true" in out
    assert "no_baseline_models_write=true" in out


def test_one_round_marks_personal_flag_only(tmp_path: Path) -> None:
    parent = tmp_path / "parent.keras"
    parent.write_bytes(b"x")
    store = MagicMock()
    store.count_unprocessed_train.side_effect = [500, 0, 0]
    store.fetch_unprocessed_train_batch.return_value = ([object(), object()], ["g1", "g2"])
    registry = MagicMock()
    registry.mark_processed.return_value = 2
    trainer = MagicMock()
    trainer.fit.return_value = (MagicMock(), {})
    metrics = _metrics()

    with (
        patch("scripts.ops.offline_user_catch_up.get_db_client", return_value=MagicMock()),
        patch(
            "scripts.ops.offline_user_catch_up.load_account_registry_bucket_datums",
            return_value=[object()] * 12,
        ),
        patch("scripts.ops.offline_user_catch_up.TrainingDataStore", return_value=store),
        patch("scripts.ops.offline_user_catch_up.get_split_registry", return_value=registry),
        patch("scripts.ops.offline_user_catch_up.HybridBoardTrainer", return_value=trainer),
        patch(
            "scripts.ops.offline_user_catch_up.load_hybrid_board_keras", return_value=MagicMock()
        ),
        patch("scripts.ops.offline_user_catch_up.evaluate_datums", return_value=metrics),
        patch("scripts.ops.offline_user_catch_up.HybridBoardTrainer.save"),
    ):
        code = run_offline_user_catch_up(
            account_id="acct-1",
            split_version="baseline-v1",
            parent_weights=str(parent),
            val_limit=None,
            max_rounds=3,
            min_new_moves=300,
            batch_limit=1000,
            epochs=1,
            style_disagree_boost=4.0,
            style_disagree_scale=2.0,
            output_dir=str(tmp_path / "out"),
        )

    assert code == 0
    trainer.fit.assert_called_once()
    assert trainer.fit.call_args.kwargs["require_parent_weights"] is True
    registry.mark_processed.assert_called_once_with(
        ["g1", "g2"],
        flag_column=PROCESSED_FLAG_PERSONAL,
    )
    fetch_kwargs = store.fetch_unprocessed_train_batch.call_args.kwargs
    assert fetch_kwargs["flag_column"] == PROCESSED_FLAG_PERSONAL
    assert "acct-1" in fetch_kwargs["extra_where"]
