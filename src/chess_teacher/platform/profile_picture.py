"""Profile picture storage: OAuth URLs, bundled assets (``asset:``), user uploads (``upload:``)."""

from __future__ import annotations

from pathlib import Path
from typing import TypeGuard

from chess_teacher.platform.account import AppLogoVariant
from chess_teacher.utils.object_storage.base import ObjectStorage
from chess_teacher.utils.object_storage.factory import get_raw_storage
from chess_teacher.utils.object_storage.images import (
    asset_image_key,
    read_asset_image,
    storage_image_data_uri,
)

_UPLOAD_PREFIX = "upload:"
_ASSET_PREFIX = "asset:"
_UPLOAD_STORAGE_PREFIX = "assets/profile_pictures"
_UPLOAD_URI_SESSION_KEY = "_chess_teacher_upload_data_uri_cache"
_PRESIGNED_URL_EXPIRES_IN = 3600
_APP_LOGO_ASSET_FILES: dict[AppLogoVariant, str] = {
    "black": "app-logo-black.svg",
    "white": "app-logo-white.svg",
}
_ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg"}


def clear_upload_image_cache(picture: str | None = None) -> None:
    """Drop cached upload data URIs from the active Streamlit session."""
    try:
        import streamlit as st
    except ImportError:
        return

    if picture is None:
        st.session_state.pop(_UPLOAD_URI_SESSION_KEY, None)
        return

    cache = st.session_state.get(_UPLOAD_URI_SESSION_KEY)
    if isinstance(cache, dict):
        cache.pop(picture, None)


class ProfilePictureService:
    """App-managed profile images stored in raw object storage under ``assets/profile_pictures/``."""

    def __init__(self, storage: ObjectStorage | None = None) -> None:
        self._storage = storage if storage is not None else get_raw_storage()

    def _object_key(self, filename: str) -> str:
        return ObjectStorage.resolve_key(_UPLOAD_STORAGE_PREFIX, filename)

    def is_upload(self, picture: str | None) -> TypeGuard[str]:
        if picture is None:
            return False
        return picture.startswith(_UPLOAD_PREFIX)

    def is_asset(self, picture: str | None) -> TypeGuard[str]:
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
        """Stable ``User.picture`` value pointing at a wordmark under ``assets/images``."""
        filename = _APP_LOGO_ASSET_FILES[variant]
        if read_asset_image(filename) is None:
            raise FileNotFoundError(f"App logo not found: {asset_image_key(filename)}")
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
        self._storage.write_bytes(self._object_key(key), data, overwrite=True)
        return f"{_UPLOAD_PREFIX}{key}"

    def delete(self, picture: str | None) -> None:
        """Remove a user upload copy; bundled assets and remote URLs are left untouched."""
        if not self.is_upload(picture):
            return
        clear_upload_image_cache(picture)
        self._storage.delete(self._object_key(self._upload_object_key(picture)))

    def _asset_filename(self, picture: str) -> str:
        if not self.is_asset(picture):
            raise ValueError(f"Not an asset picture reference: {picture!r}")
        filename = picture.removeprefix(_ASSET_PREFIX)
        if filename not in _APP_LOGO_ASSET_FILES.values():
            raise ValueError(f"Unknown asset picture: {filename!r}")
        return filename

    def _asset_data_uri(self, picture: str | None) -> str | None:
        if not self.is_asset(picture):
            return None
        filename = self._asset_filename(picture)
        return storage_image_data_uri(asset_image_key(filename))

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

    def _read_upload_bytes(self, picture: str | None) -> bytes | None:
        if not self.is_upload(picture):
            return None
        return self._storage.read_bytes(self._object_key(self._upload_object_key(picture)))

    def _session_upload_data_uri(self, picture: str) -> str | None:
        try:
            import streamlit as st
        except ImportError:
            return None

        cache = st.session_state.get(_UPLOAD_URI_SESSION_KEY)
        if isinstance(cache, dict):
            cached = cache.get(picture)
            if isinstance(cached, str):
                return cached
        return None

    def _store_session_upload_data_uri(self, picture: str, data_uri: str) -> None:
        try:
            import streamlit as st
        except ImportError:
            return

        cache = st.session_state.setdefault(_UPLOAD_URI_SESSION_KEY, {})
        if isinstance(cache, dict):
            cache[picture] = data_uri

    def _upload_data_uri(self, picture: str | None) -> str | None:
        if not self.is_upload(picture):
            return None

        cached = self._session_upload_data_uri(picture)
        if cached is not None:
            return cached

        data = self._read_upload_bytes(picture)
        if data is None:
            return None

        from chess_teacher.utils.object_storage.images import bytes_to_data_uri, mime_type_for_key

        key = self._upload_object_key(picture)
        data_uri = bytes_to_data_uri(data, mime_type_for_key(key))
        self._store_session_upload_data_uri(picture, data_uri)
        return data_uri

    def _resolve_upload_url(self, picture: str | None) -> str | None:
        """Presigned HTTPS URL for an upload when the storage backend supports it."""
        if not self.is_upload(picture):
            return None
        key = self._object_key(self._upload_object_key(picture))
        return self._storage.presigned_get_url(key, expires_in=_PRESIGNED_URL_EXPIRES_IN)


profile_pictures = ProfilePictureService()
