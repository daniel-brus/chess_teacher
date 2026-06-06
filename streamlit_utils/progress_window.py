"""Streamlit implementation of the ProgressWindow protocol."""

from __future__ import annotations

import html
import time
from dataclasses import dataclass
from types import TracebackType

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from chess_teacher.pipelines.pipeline_helpers import ProgressWindow
from streamlit_utils.layout import ingest_css

_MAX_LINES = 3

# DELAY WITH NO OTHER PURPOSE THAN PRESENTATION.
_PROGRESS_WINDOW_SLEEP_TIME: float = 0.67


_PROGRESS_CSS = """
@keyframes pw-spin{to{transform:rotate(360deg)}}
.pw-card{background:#fff;border:1px solid rgba(0,0,0,0.1);border-radius:8px;padding:10px 12px;box-shadow:0 1px 2px rgba(0,0,0,0.05);box-sizing:border-box}
.pw-wrap{box-sizing:border-box}
.pw-table{width:100%;border-collapse:collapse;border-spacing:0}
.pw-table td{vertical-align:middle;padding:0}
.pw-icon-cell{width:28px;padding-right:12px;white-space:nowrap}
.pw-spinner{width:16px;height:16px;border:2px solid rgba(0,0,0,0.12);border-top-color:rgba(0,0,0,0.45);border-radius:50%;animation:pw-spin 0.8s linear infinite;display:inline-block;vertical-align:middle}
.pw-badge{width:16px;height:16px;border-radius:50%;display:inline-block;vertical-align:middle;text-align:center;line-height:16px;font-size:10px;font-weight:600}
.pw-lines-cell{min-width:0}
.pw-lines-stack{display:block}
.pw-line{display:block;font-size:13px;line-height:1.5;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.pw-primary{color:#1f2937;font-weight:500}
.pw-secondary{color:#6b7280;font-size:12px}
.pw-success{color:#1a7f4b;font-size:13px;font-weight:500}
.pw-warning{color:#b45309}
.pw-error{color:#dc2626;font-weight:500}
"""

_ICON_SPINNER = '<div class="pw-spinner"></div>'
_ICON_SUCCESS = '<div class="pw-badge" style="background:#dcfce7;color:#1a7f4b">&#10003;</div>'
_ICON_WARNING = '<div class="pw-badge" style="background:#fef3c7;color:#b45309">&#9888;</div>'
_ICON_ERROR = '<div class="pw-badge" style="background:#fee2e2;color:#dc2626">&#10007;</div>'


@dataclass(frozen=True)
class ProgressSnapshot:
    """Frozen progress panel for display after the live run ends."""

    lines: tuple[tuple[str, str], ...]
    icon_html: str


def _inject_progress_css() -> None:
    """Inject progress styles every rerun (Streamlit rebuilds the DOM on each rerun)."""
    ingest_css(_PROGRESS_CSS)


def _build_panel_html(lines: list[tuple[str, str]], icon_html: str) -> str:
    rows = (
        "".join(f'<span class="pw-line pw-{cls}">{html.escape(msg)}</span>' for cls, msg in lines)
        if lines
        else '<span class="pw-line pw-secondary">Waiting...</span>'
    )

    return (
        '<div class="pw-card">'
        '<div class="pw-wrap">'
        '<table class="pw-table" role="presentation"><tr>'
        f'<td class="pw-icon-cell">{icon_html}</td>'
        f'<td class="pw-lines-cell"><div class="pw-lines-stack">{rows}</div></td>'
        "</tr></table></div></div>"
    )


def render_progress_snapshot(snapshot: ProgressSnapshot) -> None:
    """Render a saved pipeline progress panel (survives ``st.rerun``)."""
    _inject_progress_css()
    st.html(_build_panel_html(list(snapshot.lines), snapshot.icon_html))


def _render_panel(
    placeholder: DeltaGenerator,
    lines: list[tuple[str, str]],
    icon_html: str,
) -> None:
    panel = _build_panel_html(lines, icon_html)
    if hasattr(placeholder, "html"):
        placeholder.html(panel)
    else:
        st.markdown(panel, unsafe_allow_html=True)


class StreamlitProgressWindow(ProgressWindow):
    """
    Streamlit implementation of the ProgressWindow protocol.

    Compact spinner-style progress widget with _MAX_LINES max lines.
    No outer container — renders directly into a st.empty() placeholder.

    Usage:
        with StreamlitProgressWindow() as progress:
            progress.next("Fetching from Lichess...")
            progress.update("Fetching from Lichess... 142 found")
            progress.success("Done — 142 games saved.")
    """

    def __init__(self) -> None:
        self._lines: list[tuple[str, str]] = []
        self._icon: str = _ICON_SPINNER
        self._placeholder: DeltaGenerator | None = None
        self._final_state: str | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> ProgressWindow:
        _inject_progress_css()
        self._placeholder = st.empty()
        self._render()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if exc_type is not None and self._final_state is None:
            self.error("Pipeline failed unexpectedly.")

    # ------------------------------------------------------------------
    # ProgressWindow protocol
    # ------------------------------------------------------------------

    def next(self, message: str) -> None:
        """Add a new message. Oldest slides out if at max capacity."""
        self._icon = _ICON_SPINNER
        self._push("secondary", message)
        self._normalize_info_line_styles()
        self._render()
        time.sleep(_PROGRESS_WINDOW_SLEEP_TIME)

    def update(self, message: str) -> None:
        """Overwrite the last message in place."""
        self._icon = _ICON_SPINNER
        if self._lines:
            cls = self._lines[-1][0]
            self._lines[-1] = (cls, message)
        else:
            self._lines.append(("primary", message))
        self._normalize_info_line_styles()
        self._render()
        time.sleep(_PROGRESS_WINDOW_SLEEP_TIME)

    def success(self, message: str) -> None:
        """Show a single success message (replaces earlier progress lines)."""
        self._final_state = "complete"
        self._icon = _ICON_SUCCESS
        self._lines = [("success", message)]
        self._render()
        time.sleep(_PROGRESS_WINDOW_SLEEP_TIME)

    def warning(self, message: str) -> None:
        """Add a warning message without changing the icon."""
        self._push("warning", message)
        self._render()
        time.sleep(_PROGRESS_WINDOW_SLEEP_TIME)

    def error(self, message: str) -> None:
        """Replace all lines with a single error message."""
        self._final_state = "error"
        self._icon = _ICON_ERROR
        self._lines = [("error", message)]
        self._render()
        time.sleep(_PROGRESS_WINDOW_SLEEP_TIME)

    def pop(self, amount: int = 1) -> None:
        """Remove the last message(s)."""
        if amount <= 0 or not self._lines:
            return
        del self._lines[-min(amount, len(self._lines)) :]
        self._render()

    def clear(self) -> None:
        """Clear all messages and reset to spinner."""
        self._lines.clear()
        self._icon = _ICON_SPINNER
        self._render()

    def snapshot(self) -> ProgressSnapshot:
        """Capture the current panel for session state after the run finishes."""
        return ProgressSnapshot(lines=tuple(self._lines), icon_html=self._icon)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _push(self, cls: str, message: str) -> None:
        self._lines.append((cls, message))
        if len(self._lines) > _MAX_LINES:
            del self._lines[0]

    def _normalize_info_line_styles(self) -> None:
        """Keep first info line primary; remaining info lines secondary."""
        for idx, (cls, msg) in enumerate(self._lines):
            if cls not in ("primary", "secondary"):
                continue
            if idx == 0:
                self._lines[idx] = ("primary", msg)
            else:
                self._lines[idx] = ("secondary", msg)

    def _render(self) -> None:
        if self._placeholder is None:
            return
        _render_panel(self._placeholder, self._lines, self._icon)
