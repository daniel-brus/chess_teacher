"""Candidate-move Stockfish evals + features for the candidate-style head.

Search method (train + live must match)
---------------------------------------
All legal moves are scored with **one MultiPV search** on ``fen_before`` via
``Stockfish.get_top_moves(n_legal, num_nodes=...)`` (white POV). Prefer a fixed
``num_nodes`` budget so train/backfill/play stay aligned. SF scores are backfilled into ``candidate_evaluations`` during preprocessing
(:class:`~chess_teacher.pipelines.preprocessing.move_characteristics.candidate_evaluations.CandidateEvaluationsTransformation`
in :class:`~chess_teacher.pipelines.preprocessing.pipeline_steps.EnrichMoveCharacteristicsStep`)
or ``scripts/ops/backfill_candidate_evals.py`` for one-off historical fills.

Non-SF candidate features (train + live must match)
---------------------------------------------------
Computed on the fly from ``fen_before`` + UCI (not backfilled). Layout mirrors
``games.move_characteristics`` **move-dependent** fields (user/opponent POV
after + delta, move flags) plus piece/geometry identity. Position-only
``*_before`` / phase flags stay on the shared state vector.
Changing ``CANDIDATE_MOVE_FEAT_VERSION`` / ``MOVE_FEAT_DIM`` requires a
**cold-start**.

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
    StockfishEngine,
    fen_attack_pressure,
    fen_diagonal_openness,
    fen_hanging_value,
    fen_king_safety,
    fen_legal_moves,
    fen_mean_rank,
    fen_pawn_tension,
    fen_pin_value,
    fen_vertical_openness,
    material_balance_white_pov,
    move_created_fork,
    move_gave_check,
    move_is_capture,
    move_is_castle,
    move_is_en_passant,
    move_is_promotion,
)
from chess_teacher.utils.env_utils import get_optional_env_variable
from chess_teacher.utils.logging import get_logger

logger = get_logger()

# Fallback engine depth when calling without a node budget (rarely used).
CANDIDATE_STOCKFISH_DEPTH = 12
# Default MultiPV node budget for train / backfill. Tune for speed vs noise.
CANDIDATE_STOCKFISH_NODES = 50_000
# Live Play uses a much smaller budget - 50k MultiPV over ~20-40 legals is minutes/move.
# Override with env BASELINE_LIVE_CANDIDATE_NODES. Train/backfill stay at 50k.
LIVE_CANDIDATE_STOCKFISH_NODES = 1_000
CANDIDATE_SEARCH_METHOD = "multipv_nodes"


def live_candidate_stockfish_nodes() -> int:
    """Node budget for Play MultiPV (env override, else ``LIVE_CANDIDATE_STOCKFISH_NODES``)."""
    raw = get_optional_env_variable("BASELINE_LIVE_CANDIDATE_NODES")
    if not raw:
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
CANDIDATE_MOVE_FEAT_VERSION = 3

# Scales aligned with create_training_set TrainingDatum norms.
_EVAL_FEAT_TANH_SCALE = 5.0
_MATERIAL_TANH_SCALE = 15.0
_LEGAL_MOVES_SCALE = 50.0
_KING_SAFETY_MAX = 8.25
_KING_SAFETY_DELTA_SCALE = 4.0
_ATTACK_PRESSURE_TANH_SCALE = 8.0
_HANGING_VALUE_TANH_SCALE = 5.0
_PIN_VALUE_TANH_SCALE = 4.0
_MEAN_RANK_MAX = 1.0
_VERTICAL_OPENNESS_MAX = 8.0
_DIAGONAL_OPENNESS_MAX = 6.0
_OPENNESS_DELTA_SCALE = 1.0
_PAWN_TENSION_SCALE = 2.0
_BOARD_SPAN = 7.0

_PIECE_TYPES: tuple[chess.PieceType, ...] = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)

# Layout v3: MC move-dependent fields (user POV) + piece/geometry.
# SF from MultiPV; other metrics from fen_before → push candidate → fen_after.
CANDIDATE_MOVE_FEAT_KEYS: tuple[str, ...] = (
    # SF (user POV)
    "evaluation_after_user_pov",
    "evaluation_delta_user_pov",
    "delta_vs_best",
    # Material (user POV)
    "material_balance_after_user_pov",
    "material_balance_delta_user_pov",
    # Board-wide structure
    "vertical_openness_after",
    "vertical_openness_delta",
    "diagonal_openness_after",
    "diagonal_openness_delta",
    "pawn_tension_after",
    "pawn_tension_delta",
    # Side metrics after + delta (user / opponent)
    "user_legal_moves_after",
    "opponent_legal_moves_after",
    "user_legal_moves_delta",
    "opponent_legal_moves_delta",
    "user_king_safety_after",
    "opponent_king_safety_after",
    "user_king_safety_delta",
    "opponent_king_safety_delta",
    "user_mean_rank_after",
    "opponent_mean_rank_after",
    "user_mean_rank_delta",
    "opponent_mean_rank_delta",
    "user_attack_pressure_after",
    "opponent_attack_pressure_after",
    "user_attack_pressure_delta",
    "opponent_attack_pressure_delta",
    "user_hanging_value_after",
    "opponent_hanging_value_after",
    "user_hanging_value_delta",
    "opponent_hanging_value_delta",
    "user_pin_value_after",
    "opponent_pin_value_after",
    "user_pin_value_delta",
    "opponent_pin_value_delta",
    # Move flags (MC)
    "is_capture",
    "is_castle",
    "gave_check",
    "created_fork",
    "is_promotion",
    "is_en_passant",
    "is_recapture",
    # Identity / geometry (derive; not MC columns but needed to tell moves apart)
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
    "delta_file",
    "delta_rank",
    "move_distance_chebyshev",
)
# Never a magic literal — always len(keys). Logged as move_feat_dim so parent
# resume / Play listing reject old layouts after you edit the tuple above.
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


def _piece_one_hot(piece_type: chess.PieceType | None) -> list[float]:
    return [1.0 if piece_type == pt else 0.0 for pt in _PIECE_TYPES]


def _tanh(x: float, scale: float) -> float:
    return float(np.tanh(x / scale))


def _unit01(x: float, max_v: float) -> float:
    if max_v <= 0:
        return 0.0
    return float(np.clip(x / max_v, 0.0, 1.0))


def _div(x: float, scale: float) -> float:
    if scale == 0:
        return 0.0
    return float(x / scale)


def _position_metric_snapshot(board: chess.Board, fen: str) -> dict[str, float]:
    """White/black + board-wide metrics for one FEN (board must match fen)."""
    w_legal, b_legal = fen_legal_moves(fen)
    return {
        "material_white": float(material_balance_white_pov(fen)),
        "vertical": float(fen_vertical_openness(fen)),
        "diagonal": float(fen_diagonal_openness(fen)),
        "tension": float(fen_pawn_tension(fen)),
        "w_legal": float(w_legal),
        "b_legal": float(b_legal),
        "w_ks": float(fen_king_safety(board, chess.WHITE)),
        "b_ks": float(fen_king_safety(board, chess.BLACK)),
        "w_rank": float(fen_mean_rank(board, chess.WHITE)),
        "b_rank": float(fen_mean_rank(board, chess.BLACK)),
        "w_atk": float(fen_attack_pressure(board, chess.WHITE)),
        "b_atk": float(fen_attack_pressure(board, chess.BLACK)),
        "w_hang": float(fen_hanging_value(board, chess.WHITE)),
        "b_hang": float(fen_hanging_value(board, chess.BLACK)),
        "w_pin": float(fen_pin_value(board, chess.WHITE)),
        "b_pin": float(fen_pin_value(board, chess.BLACK)),
    }


def _user_opp(white: float, black: float, *, color_is_white: bool) -> tuple[float, float]:
    return (white, black) if color_is_white else (black, white)


def candidate_move_feature_vector(
    board: chess.Board,
    move: chess.Move,
    *,
    eval_after_user: float,
    delta_vs_best: float,
    evaluation_before_user: float | None,
    fen_before: str,
    before: dict[str, float],
    color_is_white: bool,
    opponent_move_was_capture: bool,
) -> np.ndarray:
    """Build ``MOVE_FEAT_DIM`` features for one legal candidate (board at fen_before)."""
    piece = board.piece_at(move.from_square)
    piece_type = piece.piece_type if piece is not None else None

    is_capture = move_is_capture(board, move)
    is_castle = move_is_castle(move)
    gave_check = move_gave_check(board, move)
    is_promo = move_is_promotion(move)
    is_ep = move_is_en_passant(board, move)
    is_recapture = bool(is_capture and opponent_move_was_capture)

    from_file = chess.square_file(move.from_square)
    from_rank = chess.square_rank(move.from_square)
    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)
    delta_file = to_file - from_file
    delta_rank = to_rank - from_rank
    chebyshev = max(abs(delta_file), abs(delta_rank))

    board.push(move)
    fen_after = board.fen(en_passant="fen")
    after = _position_metric_snapshot(board, fen_after)
    created_fork = move_created_fork(fen_before, fen_after, move.uci())
    board.pop()

    eval_delta_user = (
        float(eval_after_user - evaluation_before_user)
        if evaluation_before_user is not None
        else 0.0
    )

    # material_white is signed white-POV; flip for Black user.
    mat_after_user = after["material_white"] if color_is_white else -after["material_white"]
    mat_before_user = before["material_white"] if color_is_white else -before["material_white"]
    mat_delta_user = mat_after_user - mat_before_user

    def side_after_delta(key_w: str, key_b: str) -> tuple[float, float, float, float]:
        u0, o0 = _user_opp(before[key_w], before[key_b], color_is_white=color_is_white)
        u1, o1 = _user_opp(after[key_w], after[key_b], color_is_white=color_is_white)
        return u1, o1, u1 - u0, o1 - o0

    u_leg, o_leg, u_leg_d, o_leg_d = side_after_delta("w_legal", "b_legal")
    u_ks, o_ks, u_ks_d, o_ks_d = side_after_delta("w_ks", "b_ks")
    u_rk, o_rk, u_rk_d, o_rk_d = side_after_delta("w_rank", "b_rank")
    u_at, o_at, u_at_d, o_at_d = side_after_delta("w_atk", "b_atk")
    u_hg, o_hg, u_hg_d, o_hg_d = side_after_delta("w_hang", "b_hang")
    u_pn, o_pn, u_pn_d, o_pn_d = side_after_delta("w_pin", "b_pin")

    v_after, v_delta = after["vertical"], after["vertical"] - before["vertical"]
    d_after, d_delta = after["diagonal"], after["diagonal"] - before["diagonal"]
    t_after, t_delta = after["tension"], after["tension"] - before["tension"]

    feats = [
        _tanh(eval_after_user, _EVAL_FEAT_TANH_SCALE),
        _tanh(eval_delta_user, _EVAL_FEAT_TANH_SCALE),
        _tanh(delta_vs_best, _EVAL_FEAT_TANH_SCALE),
        _tanh(mat_after_user, _MATERIAL_TANH_SCALE),
        _tanh(mat_delta_user, _MATERIAL_TANH_SCALE),
        _unit01(v_after, _VERTICAL_OPENNESS_MAX),
        _div(v_delta, _OPENNESS_DELTA_SCALE),
        _unit01(d_after, _DIAGONAL_OPENNESS_MAX),
        _div(d_delta, _OPENNESS_DELTA_SCALE),
        _div(t_after, _PAWN_TENSION_SCALE),
        _div(t_delta, _PAWN_TENSION_SCALE),
        _div(u_leg, _LEGAL_MOVES_SCALE),
        _div(o_leg, _LEGAL_MOVES_SCALE),
        _div(u_leg_d, _LEGAL_MOVES_SCALE),
        _div(o_leg_d, _LEGAL_MOVES_SCALE),
        _unit01(u_ks, _KING_SAFETY_MAX),
        _unit01(o_ks, _KING_SAFETY_MAX),
        _div(u_ks_d, _KING_SAFETY_DELTA_SCALE),
        _div(o_ks_d, _KING_SAFETY_DELTA_SCALE),
        _unit01(u_rk, _MEAN_RANK_MAX),
        _unit01(o_rk, _MEAN_RANK_MAX),
        _div(u_rk_d, _MEAN_RANK_MAX),
        _div(o_rk_d, _MEAN_RANK_MAX),
        _tanh(u_at, _ATTACK_PRESSURE_TANH_SCALE),
        _tanh(o_at, _ATTACK_PRESSURE_TANH_SCALE),
        _tanh(u_at_d, _ATTACK_PRESSURE_TANH_SCALE),
        _tanh(o_at_d, _ATTACK_PRESSURE_TANH_SCALE),
        _tanh(u_hg, _HANGING_VALUE_TANH_SCALE),
        _tanh(o_hg, _HANGING_VALUE_TANH_SCALE),
        _tanh(u_hg_d, _HANGING_VALUE_TANH_SCALE),
        _tanh(o_hg_d, _HANGING_VALUE_TANH_SCALE),
        _tanh(u_pn, _PIN_VALUE_TANH_SCALE),
        _tanh(o_pn, _PIN_VALUE_TANH_SCALE),
        _tanh(u_pn_d, _PIN_VALUE_TANH_SCALE),
        _tanh(o_pn_d, _PIN_VALUE_TANH_SCALE),
        1.0 if is_capture else 0.0,
        1.0 if is_castle else 0.0,
        1.0 if gave_check else 0.0,
        1.0 if created_fork else 0.0,
        1.0 if is_promo else 0.0,
        1.0 if is_ep else 0.0,
        1.0 if is_recapture else 0.0,
        *_piece_one_hot(piece_type),
        from_file / _BOARD_SPAN,
        from_rank / _BOARD_SPAN,
        to_file / _BOARD_SPAN,
        to_rank / _BOARD_SPAN,
        delta_file / _BOARD_SPAN,
        delta_rank / _BOARD_SPAN,
        chebyshev / _BOARD_SPAN,
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
    opponent_move_was_capture: bool = False,
    evaluation_before_white: float | None = None,
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

    before = _position_metric_snapshot(board, fen_before)
    evaluation_before_user = (
        None
        if evaluation_before_white is None
        else white_to_user_pov(evaluation_before_white, color_is_white=color_is_white)
    )

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
            evaluation_before_user=evaluation_before_user,
            fen_before=fen_before,
            before=before,
            color_is_white=color_is_white,
            opponent_move_was_capture=opponent_move_was_capture,
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
    opponent_move_was_capture: bool = False,
    evaluation_before_white: float | None = None,
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

    packed = pack_candidate_tensors(
        evals,
        fen_before=fen,
        color_is_white=color_is_white,
        user_move_uci=rows[0][0],
        legal_ucis=[u for u, _, _ in rows],
        max_candidates=max_candidates,
        opponent_move_was_capture=opponent_move_was_capture,
        evaluation_before_white=evaluation_before_white,
    )
    if packed is None:
        return empty
    feats, mask, _label = packed
    ucis = [u for u, _, _ in rows]
    return ucis, feats, mask
