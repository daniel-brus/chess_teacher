"""Unit tests for logging_utils."""

import json
import logging
import sys
import tempfile
from pathlib import Path

import pytest

from chess_teacher.utils import logging_utils


@pytest.fixture
def reset_logging():
    """Reset logging state before each test."""
    root = logging.getLogger()
    root.handlers.clear()
    logging_utils._logging_configured = False
    yield
    # Cleanup after test
    root.handlers.clear()
    logging_utils._logging_configured = False


@pytest.fixture
def mock_log_dir(mocker):
    """Mock _get_log_dir to use a temporary directory."""
    temp_dir = Path(tempfile.gettempdir()) / "test_logs"
    mocker.patch.object(logging_utils, "_get_log_dir", return_value=temp_dir)
    return temp_dir


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_logging_creates_handlers(self, reset_logging, mock_log_dir):
        """Test that configure_logging adds console and file handlers."""
        root = logging.getLogger()

        logging_utils.configure_logging()

        # Verify handlers were added
        assert len(root.handlers) == 2  # console + file

    def test_configure_logging_only_runs_once(self, reset_logging, mock_log_dir):
        """Test that configure_logging doesn't run twice."""
        root = logging.getLogger()

        # First call
        logging_utils.configure_logging()
        first_handler_count = len(root.handlers)

        # Second call - should not add more handlers
        logging_utils.configure_logging()
        second_handler_count = len(root.handlers)

        assert first_handler_count == second_handler_count

    def test_configure_logging_with_custom_level(self, reset_logging, mock_log_dir):
        """Test configure_logging accepts custom level parameter."""
        root = logging.getLogger()

        logging_utils.configure_logging(level="DEBUG")

        assert root.level == logging.DEBUG

    def test_configure_logging_defaults_to_info(self, reset_logging, mock_log_dir):
        """Test configure_logging defaults to INFO level."""
        root = logging.getLogger()

        logging_utils.configure_logging()

        assert root.level == logging.INFO


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_logger(self, reset_logging, mock_log_dir):
        """Test that get_logger returns a Logger instance."""
        logger = logging_utils.get_logger("test_module")
        assert isinstance(logger, logging.Logger)

    def test_get_logger_auto_configures(self, reset_logging, mock_log_dir):
        """Test that get_logger triggers configuration."""
        logging_utils.get_logger("test_module")

        # Should have configured logging
        assert logging_utils._logging_configured

    def test_get_logger_without_name_uses_caller_module(self, reset_logging, mock_log_dir):
        """Test that get_logger without name uses caller module name."""
        logger = logging_utils.get_logger()

        # Should return a logger (name depends on call context)
        assert isinstance(logger, logging.Logger)


class TestJsonLinesFormatter:
    """Tests for _JsonLinesFormatter class."""

    @pytest.fixture(autouse=True)
    def _mock_environment(self, mocker):
        """Mock environment variable for all tests in this class."""
        mocker.patch.object(logging_utils, "get_env_variable", return_value="test")

    def test_format_returns_valid_json(self):
        """Test that format returns valid JSON string."""
        formatter = logging_utils._JsonLinesFormatter()

        # Create a mock log record
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        # Should be valid JSON
        parsed = json.loads(result)
        assert "ts" in parsed
        assert "level" in parsed
        assert "logger" in parsed
        assert "msg" in parsed

    def test_format_includes_timestamp(self):
        """Test that output includes timestamp."""
        formatter = logging_utils._JsonLinesFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        parsed = json.loads(result)

        # Should be ISO format timestamp
        ts = parsed["ts"]
        assert "T" in ts  # ISO format has T between date and time
        assert "+" in ts or "Z" in ts or "-" in ts[-5:]  # timezone info

    def test_format_includes_all_required_fields(self):
        """Test that all required fields are present."""
        formatter = logging_utils._JsonLinesFormatter()

        record = logging.LogRecord(
            name="my_logger",
            level=logging.WARNING,
            pathname="app.py",
            lineno=42,
            msg="Something happened",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["ts"] is not None
        assert parsed["level"] == "WARNING"
        assert parsed["logger"] == "my_logger"
        assert parsed["msg"] == "Something happened"

    def test_format_handles_exception_info(self):
        """Test that exception info is included when present."""
        formatter = logging_utils._JsonLinesFormatter()

        try:
            raise ValueError("test error")
        except ValueError:
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        result = formatter.format(record)
        parsed = json.loads(result)

        assert "exc_info" in parsed

    def test_format_includes_unique_log_id(self):
        """Test that each log record gets a unique log_id."""
        formatter = logging_utils._JsonLinesFormatter()

        record1 = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Message 1",
            args=(),
            exc_info=None,
        )

        record2 = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Message 2",
            args=(),
            exc_info=None,
        )

        result1 = formatter.format(record1)
        result2 = formatter.format(record2)

        parsed1 = json.loads(result1)
        parsed2 = json.loads(result2)

        # Both should have log_id
        assert "log_id" in parsed1
        assert "log_id" in parsed2
        # Each should be unique
        assert parsed1["log_id"] != parsed2["log_id"]
        # Should be valid UUID format
        import uuid

        uuid.UUID(parsed1["log_id"])
        uuid.UUID(parsed2["log_id"])

    def test_format_includes_environment(self):
        """Test that environment field is included in JSON output."""
        formatter = logging_utils._JsonLinesFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        parsed = json.loads(result)

        assert "environment" in parsed
        assert parsed["environment"] == "test"

    def test_format_environment_raises_when_missing(self, mocker):
        """Test that format raises ValueError if ENVIRONMENT is not set."""
        mocker.patch.object(
            logging_utils,
            "get_env_variable",
            side_effect=ValueError("Missing required environment variable: ENVIRONMENT"),
        )
        formatter = logging_utils._JsonLinesFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        with pytest.raises(ValueError) as exc_info:
            formatter.format(record)

        assert "Missing required environment variable: ENVIRONMENT" in str(exc_info.value)


class TestDailyFileHandler:
    """Tests for _DailyFileHandler class."""

    def test_rotation_filename_creates_date_subdirectories(self, reset_logging, mock_log_dir):
        """Test that rotation_filename creates YYYY/MM/DD subdirectories."""
        root = logging.getLogger()

        logging_utils.configure_logging(log_dir=mock_log_dir)

        # Get the file handler we just created
        file_handler = None
        for handler in root.handlers:
            if isinstance(handler, logging_utils.TimedRotatingFileHandler):
                file_handler = handler
                break

        assert file_handler is not None

        # Test rotation_filename
        default_name = str(mock_log_dir / "app.log")
        rotated_name = file_handler.rotation_filename(default_name)

        # Should be in YYYY/MM/DD format
        assert "2026/05/07" in rotated_name or "2026" in rotated_name
        assert rotated_name.endswith("app.log")

        # The date subdirectory should exist
        rotated_path = Path(rotated_name)
        assert rotated_path.parent.exists()


class TestConsoleFormatter:
    """Tests for _ConsoleFormatter class."""

    def test_format_includes_timestamp(self):
        """Test console output includes timestamp."""
        formatter = logging_utils._ConsoleFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        assert "2026-" in result  # Year should be in output
        assert "UTC" in result  # Should show UTC

    def test_format_includes_level(self):
        """Test console output includes log level."""
        formatter = logging_utils._ConsoleFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=1,
            msg="Warning",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        assert "WARNING" in result

    def test_format_includes_logger_name(self):
        """Test console output includes logger name."""
        formatter = logging_utils._ConsoleFormatter()

        record = logging.LogRecord(
            name="my_module",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        assert "my_module" in result

    def test_format_includes_message(self):
        """Test console output includes the message."""
        formatter = logging_utils._ConsoleFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Hello World",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        assert "Hello World" in result
