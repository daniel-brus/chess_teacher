"""Pytest configuration and fixtures for the chess_teacher project."""

import sys
from pathlib import Path

import pytest

# Add src/ to Python path so modules can be imported
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))


@pytest.fixture
def project_root_path():
    """Fixture providing the project root path."""
    return project_root
