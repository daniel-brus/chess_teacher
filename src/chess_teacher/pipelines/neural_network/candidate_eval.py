"""Candidate-move Stockfish evals + features for the candidate-style head.

Search method (train + live must match)
---------------------------------------
All legal moves are scored with **one MultiPV search** on ``fen_before`` via
``Stockfish.get_top_moves(n_legal, num_nodes=...)`` (white POV). Prefer a fixed
``num_nodes`` budget so train/backfill/play stay aligned. SF scores are
**backfilled** into ``candidate_evaluations``.

Non-SF candidate features (train + live must match)
---------------------------------------------------
Capture / openness / geometry / etc. are **not** backfilled. They are computed
on the fly from ``fen_before`` + UCI when packing tensors (same code path for
training batches, promotion eval, and live play). Changing
``CANDIDATE_MOVE_FEAT_VERSION`` / ``MOVE_FEAT_DIM`` requires a **cold-start**
(parent weights with a different feat dim are refused).

Delta convention (SF)
---------------------
Stored values are **white-POV** line scores in pawns. At feature time:

1. ``eval_after_user = eval_white`` if white to move else ``-eval_white``
2. ``best_user = max(eval_after_user over candidates)``
3. ``delta_vs_best = eval_after_user - best_user``  (best → 0; worse → negative)
"""

from __future__ import annotations

import json
from typing import Any

import chess
import numpy as np

from chess_teacher.utils.chess_utils import (
    PIECE_VALUES,
    StockfishEngine,
    fen_diagonal_openness,
    fen_pawn_tension,
    fen_vertical_openness,
    move_created_fork,
    move_gave_check,
    move_is_capture,
    move_is_castle,
    move_is_en_passant,
    move_is_promotion,
)
from chess_teacher.utils.logging import get_logger

logger = get_logger()

# Fallback engine depth when calling without a node budget (rarely used).
CANDIDATE_STOCKFISH_DEPTH = 12
# Default MultiPV node budget for train / backfill. Tune for speed vs noise.
CANDIDATE_STOCKFISH_NODES = 50_000
# Live Play uses a much smaller budget - 50k MultiPV over ~20-40 legals is minutes/move.
# Override with env BASELINE_LIVE_CANDIDATE_NODES. Train/backfill stay at 50k.
LIVE_CANDIDATE_STOCKFISH_NODES = 3_000
CANDIDATE_SEARCH_METHOD = "multipv_nodes"


def live_candidate_stockfish_nodes() -> int:
    """Node budget for Play MultiPV (env override, else ``LIVE_CANDIDATE_STOCKFISH_NODES``)."""
    import os

    raw = os.getenv("BASELINE_LIVE_CANDIDATE_NODES")
    if raw is None or not str(raw).strip():
        return int(LIVE_CANDIDATE_STOCKFISH_NODES)
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning(
            "Invalid BASELINE_LIVE_CANDIDATE_NODES=%r; using default %s",
            raw,
            LIVE_CANDIDATE_STOCKFISH_NODES,
        )
        return int(LIVE_CANDIDATE_STOCKFISH_NODES)


HEAD_TYPE_CANDIDATE_STYLE = "candidate_style"

# Pad legal set for Keras batching (theoretical max ~218; 128 covers practical positions).
MAX_CANDIDATES = 128

# Bump when move-feat layout changes → forces cold-start / play listing gate.
CANDIDATE_MOVE_FEAT_VERSION = 2

_EVAL_FEAT_TANH_SCALE = 5.0
_MATERIAL_TANH_SCALE = 15.0
_BOARD_SPAN = 7.0
_OPENNESS_DELTA_SCALE = 1.0
_PAWN_TENSION_DELTA_SCALE = 2.0

_PIECE_TYPES: tuple[chess.PieceType, ...] = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)

# Layout (keep stable; document for MLflow / debugging):
# 0-1   SF: tanh(eval_after_user), tanh(delta_vs_best)
# 2-6   flags: capture, castle, check, promotion, en_passant
# 7-12  mover piece one-hot (P,N,B,R,Q,K)
# 13-17 geometry: from_file, from_rank, to_file, to_rank, chebyshev (/7)
# 18-21 after-move deltas (user POV / scaled): material, vertical_open, diagonal_open, pawn_tension
# 22    created_fork
CANDIDATE_MOVE_FEAT_KEYS: tuple[str, ...] = (
    "eval_after_user_tanh",
    "delta_vs_best_tanh",
    "is_capture",
    "is_castle",
    "gave_check",
    "is_promotion",
    "is_en_passant",
    "piece_pawn",
    "piece_knight",
    "piece_bishop",
    "piece_rook",
    "piece_queen",
    "piece_king",
    "from_file",
    "from_rank",
    "to_file",
    "to_rank",
    "move_distance_chebyshev",
    "material_delta_user_tanh",
    "vertical_openness_delta",
    "diagonal_openness_delta",
    "pawn_tension_delta",
    "created_fork",
)
MOVE_FEAT_DIM = len(CANDIDATE_MOVE_FEAT_KEYS)

PAYLOAD_KEY_DEPTH = "depth"
PAYLOAD_KEY_NODES = "num_nodes"
PAYLOAD_KEY_METHOD = "method"
PAYLOAD_KEY_EVALS = "evals_white_pov"


def white_to_user_pov(eval_white: float, *, color_is_white: bool) -> float:
    return float(eval_white) if color_is_white else float(-eval_white)


def parse_candidate_evaluations(raw: Any) -> dict[str, Any] | None:
    """Normalize DB jsonb / JSON string → payload dict, or None if missing/invalid."""
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Invalid candidate_evaluations JSON string")
            return None
    if not isinstance(raw, dict):
        return None
    evals = raw.get(PAYLOAD_KEY_EVALS)
    if not isinstance(evals, dict) or not evals:
        return None
    cleaned: dict[str, float] = {}
    for uci, val in evals.items():
        try:
            cleaned[str(uci)] = float(val)
        except (TypeError, ValueError):
            continue
    if not cleaned:
        return None

    depth = raw.get(PAYLOAD_KEY_DEPTH)
    try:
        depth_i = int(depth) if depth is not None else CANDIDATE_STOCKFISH_DEPTH
    except (TypeError, ValueError):
        depth_i = CANDIDATE_STOCKFISH_DEPTH

    nodes = raw.get(PAYLOAD_KEY_NODES)
    try:
        nodes_i = int(nodes) if nodes is not None else None
    except (TypeError, ValueError):
        nodes_i = None

    method = raw.get(PAYLOAD_KEY_METHOD)
    method_s = str(method) if method is not None else None

    out: dict[str, Any] = {
        PAYLOAD_KEY_DEPTH: depth_i,
        PAYLOAD_KEY_EVALS: cleaned,
    }
    if nodes_i is not None:
        out[PAYLOAD_KEY_NODES] = nodes_i
    if method_s:
        out[PAYLOAD_KEY_METHOD] = method_s
    return out


def build_candidate_payload(
    evals_white_pov: dict[str, float],
    *,
    depth: int = CANDIDATE_STOCKFISH_DEPTH,
    num_nodes: int | None = CANDIDATE_STOCKFISH_NODES,
    method: str = CANDIDATE_SEARCH_METHOD,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        PAYLOAD_KEY_DEPTH: int(depth),
        PAYLOAD_KEY_METHOD: method,
        PAYLOAD_KEY_EVALS: {str(k): float(v) for k, v in evals_white_pov.items()},
    }
    if num_nodes is not None:
        payload[PAYLOAD_KEY_NODES] = int(num_nodes)
    return payload


def evaluate_all_legal_after(
    engine: StockfishEngine,
    fen_before: str,
    *,
    num_nodes: int | None = CANDIDATE_STOCKFISH_NODES,
) -> dict[str, float]:
    """Score every legal move on ``fen_before``; return ``{uci: eval_white_pov}``."""
    try:
        chess.Board(fen_before)
    except ValueError:
        logger.warning("Invalid fen_before for candidate evals: %s", fen_before)
        return {}
    return engine.evaluate_all_legal_moves_white_pov(fen_before, num_nodes=num_nodes)


def candidate_move_rows(
    evals_white_pov: dict[str, float],
    *,
    color_is_white: bool,
    legal_ucis: tuple[str, ...] | list[str] | None = None,
) -> list[tuple[str, float, float]]:
    """Return ``[(uci, eval_after_user, delta_vs_best), ...]`` sorted by uci."""
    if legal_ucis is not None:
        allowed = set(legal_ucis)
        items = [(u, e) for u, e in evals_white_pov.items() if u in allowed]
    else:
        items = list(evals_white_pov.items())
    if not items:
        return []

    user_evals = [(u, white_to_user_pov(e, color_is_white=color_is_white)) for u, e in items]
    best = max(ev for _, ev in user_evals)
    rows = [(u, ev, ev - best) for u, ev in user_evals]
    rows.sort(key=lambda r: r[0])
    return rows


def _material_white_pov(board: chess.Board) -> float:
    white = 0.0
    black = 0.0
    for piece in board.piece_map().values():
        value = float(PIECE_VALUES.get(piece.piece_type, 0))
        if piece.color == chess.WHITE:
            white += value
        else:
            black += value
    return white - black


def _piece_one_hot(piece_type: chess.PieceType | None) -> list[float]:
    return [1.0 if piece_type == pt else 0.0 for pt in _PIECE_TYPES]


def candidate_move_feature_vector(
    board: chess.Board,
    move: chess.Move,
    *,
    eval_after_user: float,
    delta_vs_best: float,
    fen_before: str,
    material_before_white: float,
    vertical_before: float,
    diagonal_before: float,
    tension_before: float,
) -> np.ndarray:
    """Build ``MOVE_FEAT_DIM`` features for one legal candidate (board at fen_before)."""
    color_is_white = board.turn == chess.WHITE
    piece = board.piece_at(move.from_square)
    piece_type = piece.piece_type if piece is not None else None

    is_capture = move_is_capture(board, move)
    is_castle = move_is_castle(move)
    gave_check = move_gave_check(board, move)
    is_promo = move_is_promotion(move)
    is_ep = move_is_en_passant(board, move)

    from_file = chess.square_file(move.from_square)
    from_rank = chess.square_rank(move.from_square)
    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)
    chebyshev = max(abs(to_file - from_file), abs(to_rank - from_rank))

    board.push(move)
    fen_after = board.fen(en_passant="fen")
    material_after_white = _material_white_pov(board)
    vertical_after = fen_vertical_openness(fen_after)
    diagonal_after = fen_diagonal_openness(fen_after)
    tension_after = fen_pawn_tension(fen_after)
    created_fork = move_created_fork(fen_before, fen_after, move.uci())
    board.pop()

    material_delta_white = material_after_white - material_before_white
    material_delta_user = material_delta_white if color_is_white else -material_delta_white

    feats = [
        float(np.tanh(eval_after_user / _EVAL_FEAT_TANH_SCALE)),
        float(np.tanh(delta_vs_best / _EVAL_FEAT_TANH_SCALE)),
        1.0 if is_capture else 0.0,
        1.0 if is_castle else 0.0,
        1.0 if gave_check else 0.0,
        1.0 if is_promo else 0.0,
        1.0 if is_ep else 0.0,
        *_piece_one_hot(piece_type),
        from_file / _BOARD_SPAN,
        from_rank / _BOARD_SPAN,
        to_file / _BOARD_SPAN,
        to_rank / _BOARD_SPAN,
        chebyshev / _BOARD_SPAN,
        float(np.tanh(material_delta_user / _MATERIAL_TANH_SCALE)),
        (vertical_after - vertical_before) / _OPENNESS_DELTA_SCALE,
        (diagonal_after - diagonal_before) / _OPENNESS_DELTA_SCALE,
        (tension_after - tension_before) / _PAWN_TENSION_DELTA_SCALE,
        1.0 if created_fork else 0.0,
    ]
    out = np.asarray(feats, dtype=np.float32)
    if out.shape != (MOVE_FEAT_DIM,):
        raise RuntimeError(f"move feat dim mismatch: got {out.shape}, want ({MOVE_FEAT_DIM},)")
    return out


def pack_candidate_tensors(
    evals_white_pov: dict[str, float],
    *,
    fen_before: str,
    color_is_white: bool,
    user_move_uci: str,
    legal_ucis: tuple[str, ...] | list[str] | None = None,
    max_candidates: int = MAX_CANDIDATES,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Pack padded ``(feats[MAX,F], mask[MAX], label_index)``.

    SF evals from backfill; non-SF feats computed here from ``fen_before``.
    Returns None if user move missing from evals or board/move unusable.
    """
    rows = candidate_move_rows(
        evals_white_pov,
        color_is_white=color_is_white,
        legal_ucis=legal_ucis,
    )
    if not rows:
        return None
    by_uci = {u: (ev, d) for u, ev, d in rows}
    if user_move_uci not in by_uci:
        return None

    try:
        board = chess.Board(fen_before)
    except ValueError:
        logger.warning("Invalid fen_before for candidate feats: %s", fen_before)
        return None

    if len(rows) > max_candidates:
        user_row = (user_move_uci, *by_uci[user_move_uci])
        others = [r for r in rows if r[0] != user_move_uci]
        others.sort(key=lambda r: r[1], reverse=True)
        rows = [user_row, *others[: max_candidates - 1]]
        rows.sort(key=lambda r: r[0])

    material_before = _material_white_pov(board)
    vertical_before = fen_vertical_openness(fen_before)
    diagonal_before = fen_diagonal_openness(fen_before)
    tension_before = fen_pawn_tension(fen_before)

    feats = np.zeros((max_candidates, MOVE_FEAT_DIM), dtype=np.float32)
    mask = np.zeros((max_candidates,), dtype=np.float32)
    label = -1
    for i, (uci, ev, delta) in enumerate(rows):
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            continue
        if move not in board.legal_moves:
            continue
        feats[i] = candidate_move_feature_vector(
            board,
            move,
            eval_after_user=ev,
            delta_vs_best=delta,
            fen_before=fen_before,
            material_before_white=material_before,
            vertical_before=vertical_before,
            diagonal_before=diagonal_before,
            tension_before=tension_before,
        )
        mask[i] = 1.0
        if uci == user_move_uci:
            label = i
    if label < 0 or float(mask.sum()) < 1.0:
        return None
    return feats, mask, label


def live_candidate_tensors(
    engine: StockfishEngine,
    board: chess.Board,
    *,
    max_candidates: int = MAX_CANDIDATES,
    num_nodes: int | None = CANDIDATE_STOCKFISH_NODES,
    evals: dict[str, float] | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Evaluate live legal moves; return ``(ucis, feats, mask)`` (ucis = valid slots).

    Pass precomputed ``evals`` to skip a second Stockfish MultiPV (Play path).
    """
    fen = board.fen(en_passant="fen")
    if evals is None:
        evals = evaluate_all_legal_after(engine, fen, num_nodes=num_nodes)
    color_is_white = board.turn == chess.WHITE
    legal = tuple(m.uci() for m in board.legal_moves)
    rows = candidate_move_rows(evals, color_is_white=color_is_white, legal_ucis=legal)
    empty: tuple[list[str], np.ndarray, np.ndarray] = (
        [],
        np.zeros((max_candidates, MOVE_FEAT_DIM), np.float32),
        np.zeros((max_candidates,), np.float32),
    )
    if not rows:
        return empty

    if len(rows) > max_candidates:
        rows = sorted(rows, key=lambda r: r[1], reverse=True)[:max_candidates]
        rows.sort(key=lambda r: r[0])
    else:
        rows = list(rows)

    # Label unused at play; pass first UCI so packer accepts the set.
    packed = pack_candidate_tensors(
        evals,
        fen_before=fen,
        color_is_white=color_is_white,
        user_move_uci=rows[0][0],
        legal_ucis=[u for u, _, _ in rows],
        max_candidates=max_candidates,
    )
    if packed is None:
        return empty
    feats, mask, _label = packed
    ucis = [u for u, _, _ in rows]
    return ucis, feats, mask
