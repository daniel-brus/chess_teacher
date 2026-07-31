"""Neural baseline bot: masked policy logits → legal move."""

from __future__ import annotations

import chess
import numpy as np

from chess_teacher.pipelines.neural_network.live_state import LiveStateEncoder
from chess_teacher.pipelines.neural_network.mlflow_utils import MLflowTracker
from chess_teacher.pipelines.neural_network.move_encoding import (
    POLICY_VOCAB_SIZE,
    select_move_from_logits,
)
from chess_teacher.pipelines.neural_network.train import load_policy_from_uri
from chess_teacher.utils.chess_bots.base import ChessBot
from chess_teacher.utils.logging import get_logger

logger = get_logger()


class NeuralBaselineBot(ChessBot):
    """Load policy Keras weights; pick move via masked argmax (or temperature sample)."""

    name = "NeuralBaseline"

    def __init__(
        self,
        *,
        model_uri: str,
        version: str | None = None,
        temperature: float = 0.0,
        tracker: MLflowTracker | None = None,
        state_encoder: LiveStateEncoder | None = None,
        stockfish_depth: int = LiveStateEncoder.DEFAULT_STOCKFISH_DEPTH,
    ) -> None:
        self.model_uri = model_uri
        self.version = version
        self.temperature = temperature
        self.name = f"Baseline {version}" if version else "NeuralBaseline"
        self._tracker = tracker or MLflowTracker()
        self._owns_encoder = state_encoder is None
        self._encoder = state_encoder or LiveStateEncoder(stockfish_depth=stockfish_depth)
        self._model = load_policy_from_uri(
            model_uri,
            tracker=self._tracker,
            vocab_size=POLICY_VOCAB_SIZE,
        )
        logger.info("Loaded neural baseline weights uri=%s version=%s", model_uri, version)

    def choose_move(self, board: chess.Board) -> chess.Move:
        last_uci = board.peek().uci() if board.move_stack else None
        state = self._encoder.encode(board, last_opponent_move_uci=last_uci)
        x = np.asarray(state, dtype=np.float32)[None, :]
        logits = np.asarray(self._model.predict(x, verbose=0), dtype=np.float64)[0]
        return select_move_from_logits(logits, board, temperature=self.temperature)

    def close(self) -> None:
        if self._owns_encoder:
            self._encoder.close()
        self._model = None
