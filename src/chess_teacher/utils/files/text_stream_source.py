from __future__ import annotations

from typing import NamedTuple, TextIO


class TextStreamSource(NamedTuple):
    """An open text stream and optional label for errors and record metadata."""

    stream: TextIO
    source_name: str | None = None
