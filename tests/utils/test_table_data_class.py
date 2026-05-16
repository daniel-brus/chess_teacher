"""Ensure TableDataClass dataclass fields stay in sync with metadata.yml."""

from __future__ import annotations

import pytest

from chess_teacher.platform.user import User
from chess_teacher.utils.table_data_class import TableDataClass

CLASSES_TO_TEST: list[type[TableDataClass]] = [User]


@pytest.mark.parametrize("model_cls", CLASSES_TO_TEST)
class TestMetadataSync:
    """Dataclass fields vs metadata.yml for each TableDataClass."""

    def test_metadata_in_sync(self, model_cls: type[TableDataClass]) -> None:
        errors = model_cls.validate_metadata_sync()
        assert not errors, "\n  ".join(errors)
