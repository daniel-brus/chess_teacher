"""Unit tests for env_utils."""

import os

import pytest
from chess_teacher.utils import env_utils


class TestGetEnvVariable:
    """Tests for get_env_variable function."""

    def test_get_env_variable_returns_value(self, monkeypatch):
        """Test that get_env_variable returns value from environment."""
        monkeypatch.setenv("TEST_VAR", "test_value")
        result = env_utils.get_env_variable("TEST_VAR")
        assert result == "test_value"

    def test_get_env_variable_raises_when_missing(self):
        """Test that get_env_variable raises ValueError when variable is missing."""
        # Make sure the variable doesn't exist
        os.environ.pop("NONEXISTENT_VAR", None)

        with pytest.raises(ValueError) as exc_info:
            env_utils.get_env_variable("NONEXISTENT_VAR")

        assert "Missing required environment variable: NONEXISTENT_VAR" in str(exc_info.value)

    def test_get_env_variable_uses_default(self, monkeypatch):
        """Test that get_env_variable uses default value when variable not set."""
        monkeypatch.delenv("TEST_VAR_WITH_DEFAULT", raising=False)
        result = env_utils.get_env_variable("TEST_VAR_WITH_DEFAULT", default="default_value")
        assert result == "default_value"

    def test_get_env_variable_default_none_raises(self):
        """Test that get_env_variable raises when variable missing and no default."""
        os.environ.pop("MISSING_NO_DEFAULT", None)

        with pytest.raises(ValueError):
            env_utils.get_env_variable("MISSING_NO_DEFAULT")

    def test_get_env_variable_preserves_env_var_with_spaces(self, monkeypatch):
        """Test that environment variables with spaces are preserved."""
        monkeypatch.setenv("VAR_WITH_SPACES", "value with spaces")
        result = env_utils.get_env_variable("VAR_WITH_SPACES")
        assert result == "value with spaces"

    def test_get_env_variable_preserves_multiline_env_var(self, monkeypatch):
        """Test that environment variables with newlines are preserved."""
        multiline_value = "line1\nline2\nline3"
        monkeypatch.setenv("MULTILINE_VAR", multiline_value)
        result = env_utils.get_env_variable("MULTILINE_VAR")
        assert result == multiline_value
