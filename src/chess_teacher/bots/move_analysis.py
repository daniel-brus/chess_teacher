"""Build bot move-candidate panel payloads from scored legal UCIs."""

from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

from chess_teacher.pipelines.neural_network.candidate_eval import candidate_move_rows
from chess_teacher.utils.logging import get_logger

logger = get_logger()

_TOP_EXTRA = 4  # played + this many → ≤5 rows


@dataclass(frozen=True)
class BotMoveCandidateRow:
    uci: str
    san: str
    model_p: float  # [0, 1], softmax over masked scored logits (T=1)
    sf_eval_stm: float  # pawns, bot STM POV after move (== candidate_eval user POV)
    delta_vs_best: float  # pawns, same POV (≤ 0)
    is_played: bool


@dataclass(frozen=True)
class BotMoveAnalysis:
    fen_before: str
    played_uci: str
    played_san: str
    rows: tuple[BotMoveCandidateRow, ...]  # 0..5; empty = no analysis; else played first
    temperature_used: float  # sampling T; display P always T=1 softmax


def empty_bot_move_analysis(
    board: chess.Board,
    played: chess.Move,
    *,
    temperature_used: float,
) -> BotMoveAnalysis:
    """Analysis with ``rows=()`` for empty-candidate / no-scored fallback plies."""
    fen_before = board.fen(en_passant="fen")
    try:
        played_san = board.san(played)
    except ValueError:
        played_san = played.uci()
    return BotMoveAnalysis(
        fen_before=fen_before,
        played_uci=played.uci(),
        played_san=played_san,
        rows=(),
        temperature_used=float(temperature_used),
    )


def _display_softmax_probs(logits: np.ndarray, mask: np.ndarray, n: int) -> np.ndarray:
    """Softmax over masked legal slots at temperature 1.0; illegal/pad → 0."""
    scores = np.where(mask[:n] > 0.5, np.asarray(logits[:n], dtype=np.float64), -np.inf)
    finite = np.isfinite(scores)
    probs = np.zeros(n, dtype=np.float64)
    if not np.any(finite):
        return probs
    z = scores - np.max(scores[finite])
    exp = np.exp(z)
    exp[~finite] = 0.0
    total = float(np.sum(exp))
    if total <= 0.0:
        return probs
    probs = exp / total
    return probs


def _san_for_uci(board: chess.Board, uci: str) -> str | None:
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        return None
    if move not in board.legal_moves:
        return None
    return board.san(move)


def build_bot_move_analysis(
    ucis: list[str] | tuple[str, ...],
    logits: np.ndarray,
    mask: np.ndarray,
    evals: dict[str, float],
    board: chess.Board,
    played_uci: str,
    temperature_used: float,
) -> BotMoveAnalysis:
    """Assemble ≤5 panel rows: played pinned first, then next-highest model P.

    Display probabilities use temperature-1 softmax over masked scored slots only.
    SF eval / Δ come from ``candidate_move_rows`` (bot STM POV). Missing evals
    skip that candidate (no invented 0.0).
    """
    fen_before = board.fen(en_passant="fen")
    played_san = _san_for_uci(board, played_uci) or played_uci
    n = len(ucis)
    if n == 0:
        return BotMoveAnalysis(
            fen_before=fen_before,
            played_uci=played_uci,
            played_san=played_san,
            rows=(),
            temperature_used=float(temperature_used),
        )

    probs = _display_softmax_probs(logits, mask, n)
    color_is_white = board.turn == chess.WHITE
    sf_rows = candidate_move_rows(
        evals,
        color_is_white=color_is_white,
        legal_ucis=list(ucis),
    )
    sf_by_uci = {uci: (ev, delta) for uci, ev, delta in sf_rows}

    candidates: list[BotMoveCandidateRow] = []
    for i, uci in enumerate(ucis):
        if float(mask[i]) <= 0.5:
            continue
        sf = sf_by_uci.get(uci)
        if sf is None:
            logger.warning(
                "build_bot_move_analysis: scored uci missing from evals; skip uci=%s fen=%s",
                uci,
                fen_before,
            )
            continue
        san = _san_for_uci(board, uci)
        if san is None:
            logger.warning(
                "build_bot_move_analysis: cannot SAN scored uci=%s fen=%s",
                uci,
                fen_before,
            )
            continue
        ev, delta = sf
        candidates.append(
            BotMoveCandidateRow(
                uci=uci,
                san=san,
                model_p=float(probs[i]),
                sf_eval_stm=float(ev),
                delta_vs_best=float(delta),
                is_played=uci == played_uci,
            )
        )

    played_row = next((c for c in candidates if c.uci == played_uci), None)
    if played_row is None:
        # Resilient pin: played may be masked / missing SF / bad SAN in loop above.
        played_idx = next((i for i, u in enumerate(ucis) if u == played_uci), None)
        model_p = float(probs[played_idx]) if played_idx is not None else 0.0
        sf = sf_by_uci.get(played_uci)
        if sf is not None:
            ev, delta = float(sf[0]), float(sf[1])
        else:
            logger.warning(
                "build_bot_move_analysis: played uci missing SF; emit resilient row uci=%s fen=%s",
                played_uci,
                fen_before,
            )
            ev, delta = 0.0, 0.0
        played_row = BotMoveCandidateRow(
            uci=played_uci,
            san=played_san,
            model_p=model_p,
            sf_eval_stm=ev,
            delta_vs_best=delta,
            is_played=True,
        )

    ranked = sorted(candidates, key=lambda c: c.model_p, reverse=True)
    extras = [c for c in ranked if c.uci != played_uci][:_TOP_EXTRA]
    # Ensure played flag only on pinned row.
    pinned = BotMoveCandidateRow(
        uci=played_row.uci,
        san=played_row.san,
        model_p=played_row.model_p,
        sf_eval_stm=played_row.sf_eval_stm,
        delta_vs_best=played_row.delta_vs_best,
        is_played=True,
    )
    others = tuple(
        BotMoveCandidateRow(
            uci=c.uci,
            san=c.san,
            model_p=c.model_p,
            sf_eval_stm=c.sf_eval_stm,
            delta_vs_best=c.delta_vs_best,
            is_played=False,
        )
        for c in extras
    )
    return BotMoveAnalysis(
        fen_before=fen_before,
        played_uci=played_uci,
        played_san=played_san,
        rows=(pinned, *others),
        temperature_used=float(temperature_used),
    )
