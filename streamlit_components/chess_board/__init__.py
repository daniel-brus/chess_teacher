from __future__ import annotations

from pathlib import Path
from typing import Any

import chess
import streamlit.components.v1 as components

_FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"

_chess_board_component = components.declare_component(
    "chess_teacher_board",
    path=str(_FRONTEND_DIR),
)


def _ensure_fen(board_or_fen: chess.Board | str) -> str:
    if isinstance(board_or_fen, chess.Board):
        return board_or_fen.fen(en_passant="fen")
    return str(board_or_fen)


def chess_board(
    board_or_fen: chess.Board | str,
    *,
    key: str,
    orientation: str = "white",
    height: int = 520,
    disabled: bool = False,
    last_move_uci: str | None = None,
    instance_id: int = 0,
    show_status: bool = False,
) -> dict[str, Any] | None:
    """Render an interactive chessboard with promotion chooser.

    Returns a move payload from the frontend (or ``None`` until the user moves):
    ``{"fen": "...", "uci": "e2e4", "san": "e4", "seq": 1, "instance_id": 0}``
    """
    return _chess_board_component(
        fen=_ensure_fen(board_or_fen),
        orientation="black" if orientation == "black" else "white",
        height=int(height),
        disabled=bool(disabled),
        last_move_uci=last_move_uci or "",
        instance_id=int(instance_id),
        show_status=bool(show_status),
        key=key,
        default=None,
    )
