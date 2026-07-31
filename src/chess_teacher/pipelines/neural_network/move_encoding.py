"""Fixed AlphaZero-style move vocabulary (UCI <-> index).

Action space: ``64 * 73 = 4672`` (from-square x move plane).

Planes per from-square:
- 56 queen-like slides (8 directions x distances 1..7)
- 8 knight jumps
- 9 underpromotions (3 directions x {knight, bishop, rook});
  queen promotions use the matching queen-slide plane
"""

from __future__ import annotations

import chess
import numpy as np

# Queen-like directions: N, NE, E, SE, S, SW, W, NW
_QUEEN_DIRS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
)

_KNIGHT_DELTAS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
)

# Underpromotion piece order (queen promotions use queen-slide planes).
_UNDERPROMO_PIECES: tuple[chess.PieceType, ...] = (
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
)

NUM_QUEEN_PLANES = 56
NUM_KNIGHT_PLANES = 8
NUM_UNDERPROMO_PLANES = 9
NUM_MOVE_PLANES = NUM_QUEEN_PLANES + NUM_KNIGHT_PLANES + NUM_UNDERPROMO_PLANES  # 73
POLICY_VOCAB_SIZE = 64 * NUM_MOVE_PLANES  # 4672

_QUEEN_PLANE0 = 0
_KNIGHT_PLANE0 = NUM_QUEEN_PLANES
_UNDERPROMO_PLANE0 = NUM_QUEEN_PLANES + NUM_KNIGHT_PLANES


def _on_board(file: int, rank: int) -> bool:
    return 0 <= file <= 7 and 0 <= rank <= 7


def _queen_plane(delta_file: int, delta_rank: int) -> int | None:
    if delta_file == 0 and delta_rank == 0:
        return None
    if delta_file != 0 and delta_rank != 0 and abs(delta_file) != abs(delta_rank):
        return None
    if delta_file != 0 and delta_rank == 0:
        distance = abs(delta_file)
        direction = 2 if delta_file > 0 else 6  # E or W
    elif delta_file == 0 and delta_rank != 0:
        distance = abs(delta_rank)
        direction = 0 if delta_rank > 0 else 4  # N or S
    else:
        distance = abs(delta_file)
        if delta_file > 0 and delta_rank > 0:
            direction = 1  # NE
        elif delta_file > 0 and delta_rank < 0:
            direction = 3  # SE
        elif delta_file < 0 and delta_rank < 0:
            direction = 5  # SW
        else:
            direction = 7  # NW
    if not 1 <= distance <= 7:
        return None
    return _QUEEN_PLANE0 + direction * 7 + (distance - 1)


def _knight_plane(delta_file: int, delta_rank: int) -> int | None:
    try:
        idx = _KNIGHT_DELTAS.index((delta_file, delta_rank))
    except ValueError:
        return None
    return _KNIGHT_PLANE0 + idx


def _underpromo_plane(delta_file: int, promotion: chess.PieceType) -> int | None:
    """Pawn underpromotion planes: left/straight/right x n/b/r (mover's forward)."""
    # Encode relative to white-forward (+rank). Black promotions use absolute
    # file/rank deltas from the move; left/right are file -1/0/+1 toward promo rank.
    if promotion not in _UNDERPROMO_PIECES:
        return None
    if delta_file not in (-1, 0, 1):
        return None
    dir_idx = delta_file + 1  # -1→0, 0→1, +1→2
    piece_idx = _UNDERPROMO_PIECES.index(promotion)
    return _UNDERPROMO_PLANE0 + dir_idx * 3 + piece_idx


def _plane_to_delta(plane: int) -> tuple[int, int, chess.PieceType | None]:
    """Return ``(delta_file, delta_rank, promotion_or_none)`` for a plane index."""
    if plane < _KNIGHT_PLANE0:
        direction, distance_m1 = divmod(plane - _QUEEN_PLANE0, 7)
        distance = distance_m1 + 1
        df, dr = _QUEEN_DIRS[direction]
        return df * distance, dr * distance, None
    if plane < _UNDERPROMO_PLANE0:
        return (*_KNIGHT_DELTAS[plane - _KNIGHT_PLANE0], None)
    under = plane - _UNDERPROMO_PLANE0
    dir_idx, piece_idx = divmod(under, 3)
    delta_file = dir_idx - 1
    # Rank delta resolved in decode using from-square (white +1 / black -1).
    return delta_file, 0, _UNDERPROMO_PIECES[piece_idx]


class MoveEncoder:
    """UCI ↔ fixed policy index; legal masks over ``POLICY_VOCAB_SIZE``."""

    vocab_size: int = POLICY_VOCAB_SIZE

    @staticmethod
    def encode(move: chess.Move | str) -> int:
        """Map a move to ``[0, POLICY_VOCAB_SIZE)``. Raises if geometrically unsupported."""
        if isinstance(move, str):
            move = chess.Move.from_uci(move)
        from_sq = move.from_square
        to_sq = move.to_square
        df = chess.square_file(to_sq) - chess.square_file(from_sq)
        dr = chess.square_rank(to_sq) - chess.square_rank(from_sq)

        if move.promotion is not None and move.promotion != chess.QUEEN:
            plane = _underpromo_plane(df, move.promotion)
            if plane is None:
                raise ValueError(f"Unsupported underpromotion: {move.uci()!r}")
            return from_sq * NUM_MOVE_PLANES + plane

        # Queen promotion and all non-promo moves use queen/knight planes.
        plane = _knight_plane(df, dr)
        if plane is None:
            plane = _queen_plane(df, dr)
        if plane is None:
            raise ValueError(f"Unsupported move geometry: {move.uci()!r}")
        return from_sq * NUM_MOVE_PLANES + plane

    @staticmethod
    def encode_uci(uci: str) -> int:
        return MoveEncoder.encode(uci)

    @staticmethod
    def decode(index: int) -> chess.Move:
        """Inverse of ``encode`` for a vocab index (may be illegal on a given board)."""
        if not 0 <= index < POLICY_VOCAB_SIZE:
            raise ValueError(f"Index out of range: {index}")
        from_sq, plane = divmod(index, NUM_MOVE_PLANES)
        df, dr, promo = _plane_to_delta(plane)
        from_file = chess.square_file(from_sq)
        from_rank = chess.square_rank(from_sq)

        if promo is not None:
            # Underpromotion: forward is toward rank 7 for white-side from ranks,
            # rank 0 for black-side. Infer from from_rank.
            if from_rank == 6:
                dr = 1
            elif from_rank == 1:
                dr = -1
            else:
                raise ValueError(f"Underpromo plane from non-promo rank: index={index}")
            to_file = from_file + df
            to_rank = from_rank + dr
            if not _on_board(to_file, to_rank):
                raise ValueError(f"Decoded underpromo off board: index={index}")
            return chess.Move(
                from_sq,
                chess.square(to_file, to_rank),
                promotion=promo,
            )

        to_file = from_file + df
        to_rank = from_rank + dr
        if not _on_board(to_file, to_rank):
            raise ValueError(f"Decoded move off board: index={index}")
        to_sq = chess.square(to_file, to_rank)
        # Queen promotion when pawn reaches last rank — encode stored no promo flag;
        # callers that need UCI with 'q' should set promotion when applying on a board.
        return chess.Move(from_sq, to_sq)

    @staticmethod
    def decode_uci(index: int) -> str:
        return MoveEncoder.decode(index).uci()

    @staticmethod
    def mask_from_board(board: chess.Board) -> np.ndarray:
        """Boolean mask shape ``(POLICY_VOCAB_SIZE,)`` — True = legal."""
        return MoveEncoder.mask_from_ucis(m.uci() for m in board.legal_moves)

    @staticmethod
    def mask_from_ucis(legal_ucis: list[str] | tuple[str, ...] | object) -> np.ndarray:
        mask = np.zeros(POLICY_VOCAB_SIZE, dtype=np.bool_)
        for uci in legal_ucis:
            try:
                idx = MoveEncoder.encode(str(uci))
            except ValueError:
                continue
            mask[idx] = True
        return mask

    @staticmethod
    def encode_with_promotion_on_board(board: chess.Board, move: chess.Move) -> int:
        """Encode ``move``; for last-rank pawn queen-promo, ensure promotion is set."""
        if move.promotion is None:
            piece = board.piece_at(move.from_square)
            if (
                piece is not None
                and piece.piece_type == chess.PAWN
                and chess.square_rank(move.to_square) in (0, 7)
            ):
                move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
        return MoveEncoder.encode(move)


def select_move_from_logits(
    logits: np.ndarray,
    board: chess.Board,
    *,
    temperature: float = 0.0,
) -> chess.Move:
    """Mask illegal moves, then argmax (temperature<=0) or sample."""
    flat = np.asarray(logits, dtype=np.float64).reshape(-1)
    if flat.shape[0] != POLICY_VOCAB_SIZE:
        raise ValueError(f"Expected logits of size {POLICY_VOCAB_SIZE}, got {flat.shape[0]}")
    mask = MoveEncoder.mask_from_board(board)
    if not mask.any():
        raise ValueError("No legal moves to select from")
    masked = np.where(mask, flat, -np.inf)
    if temperature is None or temperature <= 0:
        idx = int(np.argmax(masked))
    else:
        # Softmax over legal only
        legal_logits = flat[mask] / float(temperature)
        legal_logits = legal_logits - np.max(legal_logits)
        probs = np.exp(legal_logits)
        probs = probs / probs.sum()
        legal_indices = np.flatnonzero(mask)
        idx = int(np.random.choice(legal_indices, p=probs))
    move = MoveEncoder.decode(idx)
    # Attach queen promotion if needed for legality
    if move not in board.legal_moves:
        for promo in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
            candidate = chess.Move(move.from_square, move.to_square, promotion=promo)
            if candidate in board.legal_moves:
                return candidate
        # Fall back: any legal with same from-to
        for legal in board.legal_moves:
            if legal.from_square == move.from_square and legal.to_square == move.to_square:
                return legal
        raise ValueError(f"Decoded move {move.uci()!r} not legal after mask")
    return move


__all__ = [
    "NUM_MOVE_PLANES",
    "POLICY_VOCAB_SIZE",
    "MoveEncoder",
    "select_move_from_logits",
]
