from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from chess_teacher.platform.profile_picture import ProfilePictureService
from chess_teacher.utils.object_storage.filesystem import FilesystemObjectStorage


@pytest.fixture
def storage() -> FilesystemObjectStorage:
    root = Path(tempfile.mkdtemp(prefix="chess_profile_pic_test_"))
    yield FilesystemObjectStorage(root)
    shutil.rmtree(root, ignore_errors=True)


def test_save_purges_prior_upload_with_different_suffix(
    storage: FilesystemObjectStorage,
) -> None:
    service = ProfilePictureService(storage=storage)
    user_id = "user123"

    service.save(user_id=user_id, data=b"png-bytes", original_filename="photo.png")
    assert storage.read_bytes("assets/profile_pictures/user123.png") == b"png-bytes"

    picture = service.save(user_id=user_id, data=b"jpg-bytes", original_filename="photo.jpg")
    assert picture == "upload:user123.jpg"
    assert storage.read_bytes("assets/profile_pictures/user123.jpg") == b"jpg-bytes"
    assert storage.read_bytes("assets/profile_pictures/user123.png") is None
