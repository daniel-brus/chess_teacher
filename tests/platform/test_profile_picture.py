from __future__ import annotations

import shutil
import tempfile
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from chess_teacher.platform.profile_picture import (
    ProfilePictureService,
    prepare_profile_upload_bytes,
)
from chess_teacher.utils.object_storage.filesystem import FilesystemObjectStorage


@pytest.fixture
def storage() -> FilesystemObjectStorage:
    root = Path(tempfile.mkdtemp(prefix="chess_profile_pic_test_"))
    yield FilesystemObjectStorage(root)
    shutil.rmtree(root, ignore_errors=True)


def _raster_bytes(fmt: str, *, size: int = 8, color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (size, size), color).save(buffer, format=fmt)
    return buffer.getvalue()


def test_save_purges_prior_upload_with_different_suffix(
    storage: FilesystemObjectStorage,
) -> None:
    service = ProfilePictureService(storage=storage)
    user_id = "user123"
    png_bytes = _raster_bytes("PNG")
    jpg_bytes = _raster_bytes("JPEG")

    service.save(user_id=user_id, data=png_bytes, original_filename="photo.png")
    stored_png = storage.read_bytes("assets/profile_pictures/user123.png")
    assert stored_png is not None
    with Image.open(BytesIO(stored_png)) as image:
        assert image.format == "PNG"

    picture = service.save(user_id=user_id, data=jpg_bytes, original_filename="photo.jpg")
    assert picture == "upload:user123.jpg"
    stored_jpg = storage.read_bytes("assets/profile_pictures/user123.jpg")
    assert stored_jpg is not None
    with Image.open(BytesIO(stored_jpg)) as image:
        assert image.format == "JPEG"
    assert storage.read_bytes("assets/profile_pictures/user123.png") is None


def test_save_rejects_invalid_image_bytes(storage: FilesystemObjectStorage) -> None:
    service = ProfilePictureService(storage=storage)
    with pytest.raises(ValueError, match="valid PNG"):
        service.save(user_id="user123", data=b"not-an-image", original_filename="photo.png")


def test_compress_resizes_large_jpeg() -> None:
    original = _raster_bytes("JPEG", size=900)
    compressed = prepare_profile_upload_bytes(original, suffix=".jpg")
    with Image.open(BytesIO(compressed)) as image:
        assert max(image.size) <= 512
    assert len(compressed) < len(original)
