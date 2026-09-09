"""Unit tests for offline user finetune / promotion / catch-up (mocked; no Keras)."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from chess_teacher.pipelines.neural_network import offline_user
from chess_teacher.pipelines.neural_network.create_training_set import account_id_in_sql
from chess_teacher.pipelines.neural_network.eval_metrics import EvalMetrics
from chess_teacher.pipelines.neural_network.models import PROCESSED_FLAG_PERSONAL
from chess_teacher.pipelines.neural_network.splits import (
    DEFAULT_SPLIT_SALT,
    GameSplitResult,
    SplitBucket,
    SplitCounts,
)


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


def _user_split(train: list[object], val: list[object]) -> GameSplitResult:
    return GameSplitResult(
        train=tuple(train),  # type: ignore[arg-type]
        val=tuple(val),  # type: ignore[arg-type]
        test=(),
        salt=DEFAULT_SPLIT_SALT,
        counts=(
            SplitCounts(bucket=SplitBucket.TRAIN, n_games=1, n_moves=len(train)),
            SplitCounts(bucket=SplitBucket.VAL, n_games=1, n_moves=len(val)),
            SplitCounts(bucket=SplitBucket.TEST, n_games=0, n_moves=0),
        ),
    )


def _val_datums(n: int = 10) -> list[MagicMock]:
    return [MagicMock(game_id=f"v{i}", ply=i, move_nr=i) for i in range(n)]


def _train_datums(n: int, game_id: str = "t0") -> list[MagicMock]:
    return [MagicMock(game_id=game_id, ply=i, move_nr=i) for i in range(n)]


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    train: list[object],
    val: list[object],
    end_times: dict[str, datetime] | None = None,
) -> dict[str, object]:
    store = MagicMock()
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

    parent_path = MagicMock()
    parent_path.is_file.return_value = True
    eval_fn = MagicMock(return_value=_metrics())
    eval_uri = MagicMock(return_value=_metrics())
    db = MagicMock()
    split = _user_split(train, val)
    times = end_times or {}
    train_ids = list(dict.fromkeys(d.game_id for d in train))
    pending = {"n": len(train)}
    store.count_unprocessed_train.side_effect = lambda **_k: pending["n"]
    store.fetch_unprocessed_train_batch.return_value = (list(train), train_ids)
    store.fetch_split_val_datums.return_value = list(val)
    store.fetch_end_times.return_value = times
    store.fetch_for_game_ids.side_effect = AssertionError("catch-up must not hash-scan hydrate")

    registry = MagicMock()

    def _mark(game_ids: list[str], **_k: object) -> int:
        pending["n"] = 0
        return len(game_ids)

    registry.mark_processed.side_effect = _mark

    monkeypatch.setattr(offline_user, "get_db_client", lambda: db)
    monkeypatch.setattr(offline_user, "TrainingDataStore", store_cls)
    monkeypatch.setattr(offline_user, "BaselineTrainer", trainer_cls)
    monkeypatch.setattr(
        offline_user,
        "load_account_split",
        lambda _store, _account_id, limit=None: (split, times),
    )
    monkeypatch.setattr(
        offline_user,
        "linked_account_ids_for_account",
        lambda _db, _account_id: ["acct"],
    )
    monkeypatch.setattr(
        offline_user,
        "get_split_registry",
        lambda _db, split_version: registry,
    )
    monkeypatch.setattr(offline_user, "evaluate_datums", eval_fn)
    monkeypatch.setattr(offline_user, "evaluate_model_uri", eval_uri)
    monkeypatch.setattr(offline_user, "_keras_parent_path", lambda _uri: parent_path)
    monkeypatch.setattr(
        offline_user,
        "resolve_user_parent_uri",
        lambda _db, parent_uri: parent_uri or "s3://bucket/prod.keras",
    )

    return {
        "store": store,
        "trainer": trainer,
        "trainer_cls": trainer_cls,
        "eval_fn": eval_fn,
        "eval_uri": eval_uri,
        "db": db,
        "parent_path": parent_path,
        "registry": registry,
        "pending": pending,
    }


def test_source_avoids_hash_scan_and_production_catch_up() -> None:
    src = inspect.getsource(offline_user)
    assert "HashScanCursor" not in src
    assert "fetch_since" not in src
    assert "take_pending_game_ids" not in src
    assert "count_pending_moves" not in src
    assert "from chess_teacher.pipelines.neural_network.catch_up" not in src
    assert "from chess_teacher.pipelines.neural_network import catch_up" not in src
    assert "run_baseline_training_pipeline" not in src
    assert "run_baseline_promotion_pipeline" not in src
    assert "offline_eval" not in src
    assert "offline_catch_up" not in src
    assert "TrainingState" not in src
    assert "ApplyPromotionStep" not in src


def test_offline_user_weight_defaults() -> None:
    assert offline_user.DEFAULT_USER_STYLE_DISAGREE_BOOST == 2.0
    assert offline_user.DEFAULT_USER_BASELINE_DISAGREE_BOOST == 4.0
    assert offline_user.DEFAULT_USER_RECENCY_BOOST == 2.0
    ns = offline_user.build_finetune_eval_parser().parse_args(["--account-id", "acct"])
    assert ns.style_disagree_boost == 2.0
    assert ns.baseline_disagree_boost == 4.0
    assert ns.recency_boost == 2.0
    assert ns.recency_lambda == 0.02


def test_finetune_skips_fit_when_train_below_min(monkeypatch: pytest.MonkeyPatch) -> None:
    train = _train_datums(299)
    val = _val_datums(10)
    ctx = _patch_runtime(monkeypatch, train=train, val=val)
    rc = offline_user.run_offline_user_finetune_eval(
        account_id="acct",
        parent_uri="s3://bucket/parent.keras",
        min_train_moves=300,
    )
    assert rc == 0
    ctx["trainer"].fit.assert_not_called()  # type: ignore[union-attr]
    ctx["eval_uri"].assert_called_once()  # type: ignore[union-attr]
    scored_val = ctx["eval_uri"].call_args.args[1]  # type: ignore[union-attr]
    assert [d.game_id for d in scored_val] == [d.game_id for d in val]


def test_finetune_fits_once_at_min_train(monkeypatch: pytest.MonkeyPatch) -> None:
    train = _train_datums(300)
    val = _val_datums(10)
    end_times = {"t0": datetime(2026, 1, 1, tzinfo=UTC)}
    ctx = _patch_runtime(monkeypatch, train=train, val=val, end_times=end_times)
    rc = offline_user.run_offline_user_finetune_eval(
        account_id="acct",
        parent_uri="s3://bucket/parent.keras",
        min_train_moves=300,
        recency_lambda=0.02,
        style_disagree_boost=2.0,
    )
    assert rc == 0
    assert ctx["trainer"].fit.call_count == 1  # type: ignore[union-attr]
    fit_kwargs = ctx["trainer"].fit.call_args.kwargs  # type: ignore[union-attr]
    assert list(ctx["trainer"].fit.call_args.args[0]) == train  # type: ignore[union-attr]
    assert fit_kwargs["recency_lambda"] == 0.02
    assert fit_kwargs["end_time_by_game_id"] == end_times
    assert fit_kwargs["require_parent_weights"] is True
    assert fit_kwargs["baseline_weights_path"] is ctx["parent_path"]
    assert fit_kwargs["weights_path"] is ctx["parent_path"]
    assert {d.game_id for d in ctx["trainer"].fit.call_args.args[0]} <= {d.game_id for d in train}  # type: ignore[union-attr]
    assert {d.game_id for d in ctx["trainer"].fit.call_args.args[0]}.isdisjoint(  # type: ignore[union-attr]
        {d.game_id for d in val}
    )
    assert ctx["trainer_cls"].call_args.kwargs["style_disagree_boost"] == 2.0  # type: ignore[union-attr]
    assert ctx["trainer_cls"].call_args.kwargs["baseline_disagree_boost"] == 4.0  # type: ignore[union-attr]
    assert ctx["trainer_cls"].call_args.kwargs["recency_boost"] == 2.0  # type: ignore[union-attr]
    ctx["eval_fn"].assert_called()  # type: ignore[union-attr]
    scored_val = ctx["eval_fn"].call_args.args[1]  # type: ignore[union-attr]
    assert [d.game_id for d in scored_val] == [d.game_id for d in val]


def test_finetune_small_val_returns_one(monkeypatch: pytest.MonkeyPatch) -> None:
    train = _train_datums(300)
    val = _val_datums(9)
    ctx = _patch_runtime(monkeypatch, train=train, val=val)
    rc = offline_user.run_offline_user_finetune_eval(
        account_id="acct",
        parent_uri="s3://bucket/parent.keras",
        min_train_moves=300,
    )
    assert rc == 1
    ctx["trainer"].fit.assert_not_called()  # type: ignore[union-attr]


def test_finetune_missing_parent_returns_one(monkeypatch: pytest.MonkeyPatch) -> None:
    train = _train_datums(300)
    val = _val_datums(10)
    _patch_runtime(monkeypatch, train=train, val=val)

    def _no_parent(_db: object, _uri: str | None) -> str:
        raise RuntimeError("No production baseline model_uri")

    monkeypatch.setattr(offline_user, "resolve_user_parent_uri", _no_parent)
    rc = offline_user.run_offline_user_finetune_eval(
        account_id="acct",
        parent_uri=None,
        min_train_moves=300,
    )
    assert rc == 1


def test_catch_up_eval_uses_same_val_each_round(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    t1 = datetime(2026, 1, 2, tzinfo=UTC)
    t2 = datetime(2026, 1, 3, tzinfo=UTC)
    g1 = [MagicMock(game_id="g1", ply=i, move_nr=i) for i in range(20)]
    g2 = [MagicMock(game_id="g2", ply=i, move_nr=i) for i in range(20)]
    g3 = [MagicMock(game_id="g3", ply=i, move_nr=i) for i in range(20)]
    train = g1 + g2 + g3
    val = _val_datums(10)
    end_times = {"g1": t0, "g2": t1, "g3": t2}
    ctx = _patch_runtime(monkeypatch, train=train, val=val, end_times=end_times)
    store = ctx["store"]
    store.count_unprocessed_train.side_effect = [60, 40, 40, 20]  # type: ignore[union-attr]
    store.fetch_unprocessed_train_batch.side_effect = [  # type: ignore[union-attr]
        (g1, ["g1"]),
        (g2, ["g2"]),
    ]
    ctx["registry"].mark_processed.side_effect = None  # type: ignore[union-attr]
    ctx["registry"].mark_processed.return_value = 1  # type: ignore[union-attr]
    rc = offline_user.run_offline_user_catch_up(
        account_id="acct",
        parent_uri="s3://bucket/parent.keras",
        max_rounds=2,
        batch_limit=20,
        min_new_moves=10,
        min_train_moves=50,
        output_dir=tmp_path,
    )
    assert rc == 3
    assert ctx["trainer"].fit.call_count == 2  # type: ignore[union-attr]
    first_fit = ctx["trainer"].fit.call_args_list[0].kwargs  # type: ignore[union-attr]
    second_fit = ctx["trainer"].fit.call_args_list[1].kwargs  # type: ignore[union-attr]
    assert first_fit["baseline_weights_path"] is ctx["parent_path"]
    assert first_fit["weights_path"] is ctx["parent_path"]
    assert second_fit["baseline_weights_path"] is ctx["parent_path"]
    assert second_fit["weights_path"] != ctx["parent_path"]
    assert Path(second_fit["weights_path"]).name == "model.keras"
    eval_fn = ctx["eval_fn"]
    assert eval_fn.call_count == 2  # type: ignore[union-attr]
    first_val = eval_fn.call_args_list[0].args[1]  # type: ignore[union-attr]
    second_val = eval_fn.call_args_list[1].args[1]  # type: ignore[union-attr]
    assert first_val is second_val
    assert [d.game_id for d in first_val] == [d.game_id for d in val]
    store.fetch_since.assert_not_called()  # type: ignore[union-attr]
    store.fetch_split_val_datums.assert_called_once()  # type: ignore[union-attr]
    assert store.fetch_unprocessed_train_batch.call_count == 2  # type: ignore[union-attr]
    first_kwargs = store.fetch_unprocessed_train_batch.call_args_list[0].kwargs  # type: ignore[union-attr]
    assert first_kwargs["flag_column"] == PROCESSED_FLAG_PERSONAL
    assert first_kwargs["extra_where"] == account_id_in_sql(["acct"])
    fit_ids = {d.game_id for d in ctx["trainer"].fit.call_args_list[0].args[0]}  # type: ignore[union-attr]
    val_ids = {d.game_id for d in val}
    assert fit_ids == {"g1"}
    assert fit_ids.isdisjoint(val_ids)
    ctx["registry"].mark_processed.assert_any_call(  # type: ignore[union-attr]
        ["g1"], flag_column=PROCESSED_FLAG_PERSONAL
    )
    ctx["registry"].mark_processed.assert_any_call(  # type: ignore[union-attr]
        ["g2"], flag_column=PROCESSED_FLAG_PERSONAL
    )


def test_catch_up_does_not_call_fetch_since_when_caught_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = _train_datums(80, game_id="g1")
    val = _val_datums(10)
    end_times = {"g1": datetime(2026, 1, 1, tzinfo=UTC)}
    ctx = _patch_runtime(monkeypatch, train=train, val=val, end_times=end_times)
    ctx["pending"]["n"] = 80  # type: ignore[index]
    rc = offline_user.run_offline_user_catch_up(
        account_id="acct",
        parent_uri="s3://bucket/parent.keras",
        max_rounds=5,
        batch_limit=100,
        min_new_moves=500,
        min_train_moves=50,
    )
    assert rc == 0
    ctx["trainer"].fit.assert_not_called()  # type: ignore[union-attr]
    ctx["store"].fetch_since.assert_not_called()  # type: ignore[union-attr]
    ctx["store"].fetch_unprocessed_train_batch.assert_not_called()  # type: ignore[union-attr]
    ctx["registry"].mark_processed.assert_not_called()  # type: ignore[union-attr]


def test_catch_up_skip_does_not_mark(monkeypatch: pytest.MonkeyPatch) -> None:
    train = _train_datums(40)
    val = _val_datums(10)
    ctx = _patch_runtime(monkeypatch, train=train, val=val)
    ctx["pending"]["n"] = 40  # type: ignore[index]
    rc = offline_user.run_offline_user_catch_up(
        account_id="acct",
        parent_uri="s3://bucket/parent.keras",
        max_rounds=5,
        batch_limit=100,
        min_new_moves=50,
        min_train_moves=300,
    )
    assert rc == 0
    ctx["trainer"].fit.assert_not_called()  # type: ignore[union-attr]
    ctx["store"].fetch_unprocessed_train_batch.assert_not_called()  # type: ignore[union-attr]
    ctx["registry"].mark_processed.assert_not_called()  # type: ignore[union-attr]
    count_kwargs = ctx["store"].count_unprocessed_train.call_args.kwargs  # type: ignore[union-attr]
    assert count_kwargs["flag_column"] == PROCESSED_FLAG_PERSONAL
    assert count_kwargs["extra_where"] == account_id_in_sql(["acct"])


def test_training_state_and_baseline_writes_never_invoked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("must not touch TrainingState / BaselineModel writes")

    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.models.TrainingState.for_baseline",
        classmethod(_boom),
    )
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.models.TrainingState.save_to_db",
        _boom,
    )
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.models.BaselineModel.save_to_db",
        _boom,
    )
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.models.BaselineModel.save_new_to_db",
        _boom,
    )
    train = _train_datums(300)
    val = _val_datums(10)
    ctx = _patch_runtime(monkeypatch, train=train, val=val)
    assert (
        offline_user.run_offline_user_finetune_eval(
            account_id="acct",
            parent_uri="s3://bucket/parent.keras",
            min_train_moves=300,
        )
        == 0
    )
    assert (
        offline_user.run_offline_user_promotion(
            account_id="acct",
            candidate_uri="s3://bucket/cand.keras",
            train_inline=False,
            parent_uri="s3://bucket/parent.keras",
            min_train_moves=300,
        )
        == 0
    )
    ctx["trainer"].fit.assert_called_once()  # type: ignore[union-attr]
    end_times = {"t0": datetime(2026, 1, 1, tzinfo=UTC)}
    _patch_runtime(monkeypatch, train=train, val=val, end_times=end_times)
    assert (
        offline_user.run_offline_user_catch_up(
            account_id="acct",
            parent_uri="s3://bucket/parent.keras",
            max_rounds=1,
            batch_limit=300,
            min_new_moves=10,
            min_train_moves=300,
        )
        == 0
    )


def test_promotion_candidate_uri_does_not_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    train = _train_datums(300)
    val = _val_datums(10)
    ctx = _patch_runtime(monkeypatch, train=train, val=val)
    rc = offline_user.run_offline_user_promotion(
        account_id="acct",
        candidate_uri="s3://bucket/cand.keras",
        train_inline=False,
        parent_uri="s3://bucket/parent.keras",
    )
    assert rc == 0
    ctx["trainer"].fit.assert_not_called()  # type: ignore[union-attr]
    assert ctx["eval_uri"].call_count == 2  # type: ignore[union-attr]


def test_promotion_rejects_both_or_neither(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(offline_user, "get_db_client", lambda: MagicMock())
    assert (
        offline_user.run_offline_user_promotion(
            account_id="acct",
            candidate_uri=None,
            train_inline=False,
            parent_uri="s3://bucket/parent.keras",
        )
        == 1
    )
    assert (
        offline_user.run_offline_user_promotion(
            account_id="acct",
            candidate_uri="s3://bucket/cand.keras",
            train_inline=True,
            parent_uri="s3://bucket/parent.keras",
        )
        == 1
    )


def test_promotion_inline_fits_from_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    train = _train_datums(300)
    val = _val_datums(10)
    ctx = _patch_runtime(monkeypatch, train=train, val=val)
    rc = offline_user.run_offline_user_promotion(
        account_id="acct",
        candidate_uri=None,
        train_inline=True,
        parent_uri="s3://bucket/parent.keras",
        min_train_moves=300,
        recency_lambda=0.02,
    )
    assert rc == 0
    assert ctx["trainer"].fit.call_count == 1  # type: ignore[union-attr]
    assert ctx["trainer"].fit.call_args.kwargs["weights_path"] is ctx["parent_path"]  # type: ignore[union-attr]
    assert ctx["trainer"].fit.call_args.kwargs["baseline_weights_path"] is ctx["parent_path"]  # type: ignore[union-attr]
    assert ctx["trainer"].fit.call_args.kwargs["recency_lambda"] == 0.02  # type: ignore[union-attr]


def test_catch_up_trains_complete_fetched_batch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    g1 = [MagicMock(game_id="g1", ply=i, move_nr=i) for i in range(12)]
    val = _val_datums(10)
    end_times = {"g1": datetime(2026, 1, 1, tzinfo=UTC)}
    ctx = _patch_runtime(monkeypatch, train=g1, val=val, end_times=end_times)
    ctx["store"].count_unprocessed_train.side_effect = [28, 8]  # type: ignore[union-attr]
    ctx["store"].fetch_unprocessed_train_batch.return_value = (g1, ["g1"])  # type: ignore[union-attr]
    ctx["registry"].mark_processed.side_effect = None  # type: ignore[union-attr]
    ctx["registry"].mark_processed.return_value = 1  # type: ignore[union-attr]
    rc = offline_user.run_offline_user_catch_up(
        account_id="acct",
        parent_uri="s3://bucket/parent.keras",
        max_rounds=1,
        batch_limit=10,
        min_new_moves=5,
        min_train_moves=20,
        output_dir=tmp_path,
    )
    assert rc == 3
    fit_batch = list(ctx["trainer"].fit.call_args.args[0])  # type: ignore[union-attr]
    assert {d.game_id for d in fit_batch} == {"g1"}
    assert len(fit_batch) == 12
    ctx["registry"].mark_processed.assert_called_once_with(  # type: ignore[union-attr]
        ["g1"], flag_column=PROCESSED_FLAG_PERSONAL
    )


def test_parser_has_no_hidden_or_promote_or_prod_cutoff() -> None:
    help_text = offline_user.build_catch_up_parser().format_help()
    assert "--hidden" not in help_text
    assert "--promote" not in help_text
    assert "--start-from-production-cutoff" not in help_text
    assert "--start-cutoff" not in help_text
    assert "--account-id" in help_text
    assert "--baseline-disagree-boost" in help_text
    assert "--recency-boost" in help_text
    promo = offline_user.build_promotion_parser().format_help()
    assert "--baseline-uri" in promo
    assert "--train-inline" in promo
    assert "--candidate-uri" in promo


def test_catch_up_skip_small_val_returns_one(monkeypatch: pytest.MonkeyPatch) -> None:
    train = _train_datums(40)
    val = _val_datums(9)
    ctx = _patch_runtime(monkeypatch, train=train, val=val)
    rc = offline_user.run_offline_user_catch_up(
        account_id="acct",
        parent_uri="s3://bucket/parent.keras",
        max_rounds=5,
        batch_limit=100,
        min_new_moves=50,
        min_train_moves=300,
    )
    assert rc == 1
    ctx["trainer"].fit.assert_not_called()  # type: ignore[union-attr]
    ctx["registry"].mark_processed.assert_not_called()  # type: ignore[union-attr]
    ctx["store"].count_unprocessed_train.assert_not_called()  # type: ignore[union-attr]


def test_linked_account_ids_resolves_user_level(monkeypatch: pytest.MonkeyPatch) -> None:
    account = MagicMock(account_id="cli-acct")
    user = MagicMock()
    user.get_linked_accounts.return_value = [
        MagicMock(account_id="cli-acct"),
        MagicMock(account_id="linked-acct"),
    ]
    monkeypatch.setattr(
        offline_user.Account,
        "fetch_from_db",
        staticmethod(lambda _db, id=None: account),
    )
    monkeypatch.setattr(
        offline_user.UserAccount,
        "fetch_all_from_db",
        staticmethod(lambda _db, where=None, **_k: [MagicMock(user_id="user-1")]),
    )
    monkeypatch.setattr(
        offline_user.User,
        "fetch_from_db",
        staticmethod(lambda _db, id=None: user),
    )
    ids = offline_user.linked_account_ids_for_account(MagicMock(), "cli-acct")
    assert ids == ["cli-acct", "linked-acct"]
    user.get_linked_accounts.assert_called_once()


def test_linked_account_ids_empty_when_unlinked(monkeypatch: pytest.MonkeyPatch) -> None:
    account = MagicMock(account_id="orphan")
    monkeypatch.setattr(
        offline_user.Account,
        "fetch_from_db",
        staticmethod(lambda _db, id=None: account),
    )
    monkeypatch.setattr(
        offline_user.UserAccount,
        "fetch_all_from_db",
        staticmethod(lambda *_a, **_k: []),
    )
    assert offline_user.linked_account_ids_for_account(MagicMock(), "orphan") == []
    assert account_id_in_sql([]) == "1 = 0"


def test_catch_up_empty_linked_accounts_matches_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = _train_datums(80)
    val = _val_datums(10)
    ctx = _patch_runtime(monkeypatch, train=train, val=val)
    monkeypatch.setattr(offline_user, "linked_account_ids_for_account", lambda *_a, **_k: [])
    ctx["pending"]["n"] = 0  # type: ignore[index]
    rc = offline_user.run_offline_user_catch_up(
        account_id="acct",
        parent_uri="s3://bucket/parent.keras",
        max_rounds=3,
        batch_limit=100,
        min_new_moves=50,
    )
    assert rc == 0
    extra = ctx["store"].count_unprocessed_train.call_args.kwargs["extra_where"]  # type: ignore[union-attr]
    assert extra == "1 = 0"
    ctx["store"].fetch_unprocessed_train_batch.assert_not_called()  # type: ignore[union-attr]
    ctx["registry"].mark_processed.assert_not_called()  # type: ignore[union-attr]


def test_catch_up_save_failure_does_not_mark(monkeypatch: pytest.MonkeyPatch) -> None:
    train = _train_datums(80)
    val = _val_datums(10)
    ctx = _patch_runtime(monkeypatch, train=train, val=val)
    ctx["pending"]["n"] = 80  # type: ignore[index]
    ctx["trainer_cls"].save = staticmethod(  # type: ignore[union-attr]
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full"))
    )
    with pytest.raises(OSError, match="disk full"):
        offline_user.run_offline_user_catch_up(
            account_id="acct",
            parent_uri="s3://bucket/parent.keras",
            max_rounds=2,
            batch_limit=80,
            min_new_moves=10,
        )
    ctx["registry"].mark_processed.assert_not_called()  # type: ignore[union-attr]


def test_finetune_incompatible_parent_returns_one(monkeypatch: pytest.MonkeyPatch) -> None:
    train = _train_datums(300)
    val = _val_datums(10)
    ctx = _patch_runtime(monkeypatch, train=train, val=val)
    ctx["trainer"].fit.side_effect = RuntimeError("Parent weights not candidate_style-compatible")  # type: ignore[union-attr]
    rc = offline_user.run_offline_user_finetune_eval(
        account_id="acct",
        parent_uri="s3://bucket/parent.keras",
        min_train_moves=300,
    )
    assert rc == 1
