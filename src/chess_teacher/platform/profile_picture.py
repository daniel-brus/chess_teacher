"""Profile picture storage: OAuth URLs, bundled assets (``asset:``), user uploads (``upload:``)."""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO

from chess_teacher.platform.account import AppLogoVariant, app_logo_path, platform_logo_images_dir
from chess_teacher.platform.user import User
from chess_teacher.utils.db_client import DatabaseClient
from chess_teacher.utils.env_utils import get_env_variable
from chess_teacher.utils.exception_utils import ConfigError

_UPLOAD_PREFIX = "upload:"
_ASSET_PREFIX = "asset:"
_APP_LOGO_ASSET_FILES: dict[AppLogoVariant, str] = {
    "black": "app-logo-black.svg",
    "white": "app-logo-white.svg",
}
_ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg"}


class _ProfilePictureStore(ABC):
    @abstractmethod
    def save(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def read(self, key: str) -> bytes | None: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...


class _FilesystemProfilePictureStore(_ProfilePictureStore):
    def __init__(self, root: Path) -> None:
        self._root = root

    def _path(self, key: str) -> Path:
        return self._root / key

    def save(self, key: str, data: bytes) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        self._path(key).write_bytes(data)

    def read(self, key: str) -> bytes | None:
        path = self._path(key)
        return path.read_bytes() if path.is_file() else None

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)


class ProfilePictureService:
    """App-managed profile images (disk today; inject another ``_ProfilePictureStore`` for blob)."""

    def __init__(self, store: _ProfilePictureStore | None = None) -> None:
        self._store = store if store is not None else self._default_store()

    @staticmethod
    def _default_store() -> _ProfilePictureStore:
        raw_dir = get_env_variable("RAW_DIR")
        if not raw_dir:
            raise ConfigError("RAW_DIR environment variable is not set")
        root = Path(raw_dir) / "assets" / "profile_pictures"
        return _FilesystemProfilePictureStore(root)

    def is_upload(self, picture: str | None) -> bool:
        if picture is None:
            return False
        return picture.startswith(_UPLOAD_PREFIX)

    def is_asset(self, picture: str | None) -> bool:
        if picture is None:
            return False
        return picture.startswith(_ASSET_PREFIX)

    def is_svg_picture(self, picture: str | None) -> bool:
        if not picture:
            return False
        if self.is_asset(picture):
            return True
        if self.is_upload(picture):
            return Path(self._upload_object_key(picture)).suffix.lower() == ".svg"
        if self.is_remote(picture):
            return picture.split("?", 1)[0].lower().endswith(".svg")
        return False

    def is_remote(self, picture: str | None) -> bool:
        return picture is not None and picture.startswith(("http://", "https://"))

    def app_logo_picture_ref(self, *, variant: AppLogoVariant) -> str:
        """Stable ``User.picture`` value pointing at a bundled wordmark under ``assets/images``."""
        filename = _APP_LOGO_ASSET_FILES[variant]
        path = app_logo_path(variant=variant)
        if not path.is_file():
            raise FileNotFoundError(f"App logo not found: {path}")
        return f"{_ASSET_PREFIX}{filename}"

    def picture_img_src(self, picture: str | None) -> str | None:
        """Value for ``<img src>``: provider HTTPS URL, asset/upload data URI, or blob URL."""
        if not picture:
            return None
        if self.is_remote(picture):
            return picture
        if self.is_asset(picture):
            return self._asset_data_uri(picture)
        if not self.is_upload(picture):
            return None
        blob_url = self._resolve_upload_url(picture)
        if blob_url:
            return blob_url
        return self._upload_data_uri(picture)

    def save(self, *, user_id: str, data: bytes, original_filename: str) -> str:
        """Persist upload; return URL value for ``User.picture``."""
        suffix = self._normalize_upload_suffix(original_filename)
        key = f"{user_id}{suffix}"
        self._store.save(key, data)
        return f"{_UPLOAD_PREFIX}{key}"

    def delete(self, picture: str | None) -> None:
        """Remove a user upload copy; bundled assets and remote URLs are left untouched."""
        if not self.is_upload(picture):
            return
        self._store.delete(self._upload_object_key(picture))  # type: ignore[arg-type]

    def _asset_filename(self, picture: str) -> str:
        if not self.is_asset(picture):
            raise ValueError(f"Not an asset picture reference: {picture!r}")
        filename = picture.removeprefix(_ASSET_PREFIX)
        if filename not in _APP_LOGO_ASSET_FILES.values():
            raise ValueError(f"Unknown asset picture: {filename!r}")
        return filename

    def _read_asset_bytes(self, picture: str | None) -> bytes | None:
        if not self.is_asset(picture):
            return None
        path = platform_logo_images_dir() / self._asset_filename(picture)  # type: ignore[arg-type]
        return path.read_bytes() if path.is_file() else None

    def _asset_data_uri(self, picture: str | None) -> str | None:
        data = self._read_asset_bytes(picture)
        if data is None:
            return None
        filename = self._asset_filename(picture)  # type: ignore[arg-type]
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{self._mime_type_for_key(filename)};base64,{encoded}"

    def _upload_object_key(self, picture: str) -> str:
        if picture.startswith(_UPLOAD_PREFIX):
            return picture.removeprefix(_UPLOAD_PREFIX)
        raise ValueError(f"Not an upload picture reference: {picture!r}")

    @staticmethod
    def _normalize_upload_suffix(filename: str) -> str:
        suffix = Path(filename).suffix.lower()
        if suffix not in _ALLOWED_SUFFIXES:
            raise ValueError(f"Unsupported image type: {suffix or '(none)'}")
        return ".jpg" if suffix == ".jpeg" else suffix

    @staticmethod
    def _mime_type_for_key(key: str) -> str:
        suffix = Path(key).suffix.lower()
        if suffix == ".png":
            return "image/png"
        if suffix in {".jpg", ".jpeg"}:
            return "image/jpeg"
        if suffix == ".webp":
            return "image/webp"
        if suffix == ".gif":
            return "image/gif"
        if suffix == ".svg":
            return "image/svg+xml"
        return "application/octet-stream"

    def _read_upload_bytes(self, picture: str | None) -> bytes | None:
        if not self.is_upload(picture):
            return None
        return self._store.read(self._upload_object_key(picture))  # type: ignore[arg-type]

    def _upload_data_uri(self, picture: str | None) -> str | None:
        data = self._read_upload_bytes(picture)
        if data is None:
            return None
        key = self._upload_object_key(picture)  # type: ignore[arg-type]
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{self._mime_type_for_key(key)};base64,{encoded}"

    def _resolve_upload_url(self, picture: str | None) -> str | None:
        """Public URL for an upload (``None`` until a blob CDN is wired)."""
        _ = picture
        return None


profile_pictures = ProfilePictureService()


def _read_upload_data(data: bytes | BinaryIO) -> bytes:
    if isinstance(data, bytes):
        return data
    return data.read()


def replace_user_profile_picture(
    user: User,
    db_client: DatabaseClient,
    *,
    data: bytes | BinaryIO,
    original_filename: str,
) -> User:
    """Replace the user's avatar with an uploaded image; persist to storage and DB."""
    upload_bytes = _read_upload_data(data)
    profile_pictures.delete(user.picture)
    picture_url = profile_pictures.save(
        user_id=user.user_id,
        data=upload_bytes,
        original_filename=original_filename,
    )
    user.upsert_field(db_client, "picture", picture_url)
    user.picture = picture_url
    return user


def replace_user_profile_picture_with_app_logo(
    user: User,
    db_client: DatabaseClient,
    *,
    variant: AppLogoVariant,
) -> User:
    """Point the user's avatar at a bundled black/white wordmark (no copy into uploads)."""
    profile_pictures.delete(user.picture)
    picture_ref = profile_pictures.app_logo_picture_ref(variant=variant)
    user.upsert_field(db_client, "picture", picture_ref)
    user.picture = picture_ref
    return user
