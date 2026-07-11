"""Pytest configuration and fixtures for the chess_teacher project."""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# Add src/ to Python path so modules can be imported
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

# chess_teacher configures logging at import time (needs STORAGE_ROOT + ENVIRONMENT)
os.environ.setdefault("STORAGE_ROOT", str(Path(tempfile.gettempdir()) / "chess_teacher_test"))
os.environ.setdefault("LOG_SHIP_ENABLED", "false")
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault(
    "LOG_BUFFER_DIR", str(Path(tempfile.gettempdir()) / "chess_teacher_test_logs")
)
os.environ.setdefault("HOSTNAME", "test-host")


@pytest.fixture(autouse=True)
def isolate_raw_storage(monkeypatch: pytest.MonkeyPatch):
    """Route ``get_raw_storage()`` to a temp local backend (no real S3 in tests)."""
    from chess_teacher.utils.object_storage import factory
    from chess_teacher.utils.object_storage.filesystem import FilesystemObjectStorage

    root = Path(tempfile.mkdtemp(prefix="chess_teacher_raw_test_"))
    storage = FilesystemObjectStorage(root)
    monkeypatch.setattr(factory, "_raw_storage", storage)
    yield storage
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def project_root_path():
    """Fixture providing the project root path."""
    return project_root
