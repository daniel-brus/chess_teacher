"""Pytest configuration and fixtures for the chess_teacher project."""

import os
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
os.environ.setdefault("STORAGE_BACKEND", "filesystem")
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault(
    "LOG_BUFFER_DIR", str(Path(tempfile.gettempdir()) / "chess_teacher_test_logs")
)
os.environ.setdefault("HOSTNAME", "test-host")


@pytest.fixture
def project_root_path():
    """Fixture providing the project root path."""
    return project_root
