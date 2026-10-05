"""MLflow tracking DB upgrade helpers + ops entrypoint."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from chess_teacher.pipelines.neural_network import mlflow_utils
from scripts.ops import mlflow_db_upgrade
from scripts.utils.run_script_job import resolve_script_relpath


def test_resolve_script_relpath_mlflow_db_upgrade() -> None:
    assert resolve_script_relpath("mlflow_db_upgrade") == "ops/mlflow_db_upgrade.py"


def test_resolve_mlflow_tracking_uri_prefers_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "postgresql://env/db")
    monkeypatch.setattr(
        mlflow_utils,
        "postgres_url_string",
        lambda: "postgresql://fallback/db",
    )
    assert (
        mlflow_utils.resolve_mlflow_tracking_uri("postgresql://explicit/db")
        == "postgresql://explicit/db"
    )
    assert mlflow_utils.resolve_mlflow_tracking_uri() == "postgresql://env/db"


def test_mlflow_tracking_schema_status(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = MagicMock()
    monkeypatch.setattr(
        mlflow_utils,
        "resolve_mlflow_tracking_uri",
        lambda tracking_uri=None: "postgresql://t/db",
    )

    import mlflow.store.db.utils as store_utils
    import sqlalchemy

    monkeypatch.setattr(sqlalchemy, "create_engine", MagicMock(return_value=engine))
    monkeypatch.setattr(store_utils, "_get_schema_version", lambda _engine: "6f8d9c3b2a1e")
    monkeypatch.setattr(store_utils, "_get_alembic_config", lambda _uri: MagicMock())

    script_dir = MagicMock()
    script_dir.get_current_head.return_value = "b7e2c1a4d9f3"
    monkeypatch.setattr(
        "alembic.script.ScriptDirectory.from_config",
        MagicMock(return_value=script_dir),
    )

    current, head = mlflow_utils.mlflow_tracking_schema_status()
    assert current == "6f8d9c3b2a1e"
    assert head == "b7e2c1a4d9f3"
    engine.dispose.assert_called_once()


def test_upgrade_mlflow_tracking_db(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = MagicMock()
    monkeypatch.setattr(
        mlflow_utils,
        "resolve_mlflow_tracking_uri",
        lambda tracking_uri=None: "postgresql://t/db",
    )

    import mlflow.store.db.utils as store_utils
    import sqlalchemy

    monkeypatch.setattr(sqlalchemy, "create_engine", MagicMock(return_value=engine))
    versions = iter(["6f8d9c3b2a1e", "b7e2c1a4d9f3"])
    monkeypatch.setattr(store_utils, "_get_schema_version", lambda _engine: next(versions))
    upgraded = MagicMock()
    monkeypatch.setattr(store_utils, "_upgrade_db", upgraded)

    before, after = mlflow_utils.upgrade_mlflow_tracking_db()
    assert before == "6f8d9c3b2a1e"
    assert after == "b7e2c1a4d9f3"
    upgraded.assert_called_once_with(engine)
    engine.dispose.assert_called_once()


def test_ops_dry_run_skips_upgrade(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        mlflow_db_upgrade,
        "resolve_mlflow_tracking_uri",
        lambda tracking_uri=None: "postgresql://user:secret@host/db",
    )
    monkeypatch.setattr(
        mlflow_db_upgrade,
        "_log_tracking_uri",
        lambda uri: "postgresql://user:***@host/db",
    )
    monkeypatch.setattr(
        mlflow_db_upgrade,
        "mlflow_tracking_schema_status",
        lambda uri: ("6f8d9c3b2a1e", "b7e2c1a4d9f3"),
    )
    upgrade = MagicMock()
    monkeypatch.setattr(mlflow_db_upgrade, "upgrade_mlflow_tracking_db", upgrade)
    monkeypatch.setattr("sys.argv", ["mlflow_db_upgrade.py", "--dry-run"])

    assert mlflow_db_upgrade.main() == 0
    upgrade.assert_not_called()
    out = capsys.readouterr().out
    assert "current=6f8d9c3b2a1e" in out
    assert "head=b7e2c1a4d9f3" in out
    assert "would_upgrade=true" in out


def test_ops_already_at_head_is_noop(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        mlflow_db_upgrade,
        "resolve_mlflow_tracking_uri",
        lambda tracking_uri=None: "postgresql://t/db",
    )
    monkeypatch.setattr(mlflow_db_upgrade, "_log_tracking_uri", lambda uri: uri)
    monkeypatch.setattr(
        mlflow_db_upgrade,
        "mlflow_tracking_schema_status",
        lambda uri: ("b7e2c1a4d9f3", "b7e2c1a4d9f3"),
    )
    upgrade = MagicMock()
    monkeypatch.setattr(mlflow_db_upgrade, "upgrade_mlflow_tracking_db", upgrade)
    monkeypatch.setattr("sys.argv", ["mlflow_db_upgrade.py"])

    assert mlflow_db_upgrade.main() == 0
    upgrade.assert_not_called()
    assert "already_at_head=true" in capsys.readouterr().out
