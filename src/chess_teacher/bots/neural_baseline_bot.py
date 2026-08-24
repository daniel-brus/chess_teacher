"""Neural baseline bot: candidate SF features → style scorer → legal move."""

from __future__ import annotations

import time

import chess
import numpy as np

from chess_teacher.bots.base import ChessBot
from chess_teacher.pipelines.neural_network.candidate_eval import (
    CANDIDATE_STOCKFISH_DEPTH,
    evaluate_all_legal_after,
    live_candidate_stockfish_nodes,
    live_candidate_tensors,
)
from chess_teacher.pipelines.neural_network.live_state import LiveStateEncoder
from chess_teacher.pipelines.neural_network.mlflow_utils import MLflowTracker
from chess_teacher.pipelines.neural_network.train import load_candidate_style_from_uri
from chess_teacher.utils.chess_utils import StockfishEngine
from chess_teacher.utils.logging import get_logger

logger = get_logger()


def _root_eval_white_pov_from_candidates(
    evals: dict[str, float],
    *,
    color_is_white: bool,
) -> float | None:
    """Approx current-position white-POV eval from MultiPV after-move scores."""
    if not evals:
        return None
    values = list(evals.values())
    # Best line for STM ≈ root eval (white POV): max if White, min if Black.
    return float(max(values) if color_is_white else min(values))


class NeuralBaselineBot(ChessBot):
    """Load candidate_style Keras weights; score live legal moves with SF feats."""

    name = "NeuralBaseline"

    def __init__(
        self,
        *,
        model_uri: str,
        version: str | None = None,
        temperature: float = 0.0,
        tracker: MLflowTracker | None = None,
        stockfish_depth: int = CANDIDATE_STOCKFISH_DEPTH,
        candidate_nodes: int | None = None,
        engine: StockfishEngine | None = None,
        state_encoder: LiveStateEncoder | None = None,
    ) -> None:
        self.model_uri = model_uri
        self.version = version
        self.temperature = temperature
        self.candidate_nodes = (
            int(candidate_nodes)
            if candidate_nodes is not None
            else live_candidate_stockfish_nodes()
        )
        self.name = f"Baseline {version}" if version else "NeuralBaseline"
        self._tracker = tracker or MLflowTracker()

        self._owns_engine = engine is None
        self._engine = engine or StockfishEngine(depth=stockfish_depth)
        if self._owns_engine:
            self._engine.__enter__()

        self._owns_encoder = state_encoder is None
        self._encoder = state_encoder or LiveStateEncoder(engine=self._engine)
        self._model = load_candidate_style_from_uri(model_uri, tracker=self._tracker)
        logger.info(
            "Loaded candidate_style baseline uri=%s version=%s depth=%s live_nodes=%s",
            model_uri,
            version,
            stockfish_depth,
            self.candidate_nodes,
        )

    def choose_move(self, board: chess.Board) -> chess.Move:
        if board.is_game_over():
            raise ValueError("Cannot choose a move in a finished game.")

        t0 = time.perf_counter()
        fen = board.fen(en_passant="fen")
        color_is_white = board.turn == chess.WHITE

        # One MultiPV at live node budget (not train/backfill 50k).
        evals = evaluate_all_legal_after(self._engine, fen, num_nodes=self.candidate_nodes)
        t_sf = time.perf_counter()

        last_uci = board.peek().uci() if board.move_stack else None
        root_eval_white = _root_eval_white_pov_from_candidates(evals, color_is_white=color_is_white)
        opponent_move_was_capture = False
        if board.move_stack:
            probe = board.copy(stack=True)
            last = probe.pop()
            opponent_move_was_capture = probe.is_capture(last)

        state = self._encoder.encode(
            board,
            last_opponent_move_uci=last_uci,
            evaluation_white_pov=root_eval_white,
        )
        t_state = time.perf_counter()

        ucis, feats, mask = live_candidate_tensors(
            self._engine,
            board,
            num_nodes=self.candidate_nodes,
            evals=evals,
            opponent_move_was_capture=opponent_move_was_capture,
            evaluation_before_white=root_eval_white,
        )
        t_feats = time.perf_counter()
        if not ucis:
            logger.warning(
                "Baseline choose_move: empty candidates; first legal nodes=%s fen=%s",
                self.candidate_nodes,
                fen,
            )
            return next(iter(board.legal_moves))

        x_state = np.asarray(state, dtype=np.float32)[None, :]
        x_feats = feats[None, :, :]
        logits = np.asarray(
            self._model.predict({"state": x_state, "move_feats": x_feats}, verbose=0),
            dtype=np.float64,
        )[0]
        n = len(ucis)
        scores = np.where(mask[:n] > 0.5, logits[:n], -np.inf)
        if self.temperature and self.temperature > 0:
            z = scores / float(self.temperature)
            z = z - np.max(z)
            p = np.exp(z)
            p = p / np.sum(p)
            idx = int(np.random.choice(n, p=p))
        else:
            idx = int(np.argmax(scores))
        move = chess.Move.from_uci(ucis[idx])
        if move not in board.legal_moves:
            raise RuntimeError(f"Candidate-style pick {ucis[idx]!r} not legal")

        t_end = time.perf_counter()
        logger.info(
            "Baseline choose_move version=%s nodes=%s n_cand=%s "
            "sf_s=%.2f state_s=%.2f feats_s=%.2f predict_s=%.2f total_s=%.2f pick=%s",
            self.version,
            self.candidate_nodes,
            n,
            t_sf - t0,
            t_state - t_sf,
            t_feats - t_state,
            t_end - t_feats,
            t_end - t0,
            move.uci(),
        )
        return move

    def close(self) -> None:
        if self._owns_encoder:
            if not self._owns_engine:
                self._encoder.close()
            else:
                self._encoder.close()
        if self._owns_engine:
            self._engine.__exit__(None, None, None)
            self._owns_engine = False
        self._model = None
