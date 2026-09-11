from __future__ import annotations

import pytest

from chess_teacher.platform.user import MAX_DISPLAY_NAME_LENGTH, normalize_display_name


def test_normalize_display_name_strips_and_collapses_whitespace() -> None:
    assert normalize_display_name("  Ada   Lovelace  ") == "Ada Lovelace"
    assert normalize_display_name("   ") is None
    assert normalize_display_name(None) is None


def test_normalize_display_name_rejects_control_characters() -> None:
    with pytest.raises(ValueError, match="control characters"):
        normalize_display_name("Ada\nLovelace")


def test_normalize_display_name_rejects_too_long() -> None:
    with pytest.raises(ValueError, match="characters or fewer"):
        normalize_display_name("n" * (MAX_DISPLAY_NAME_LENGTH + 1))
