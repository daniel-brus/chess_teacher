from __future__ import annotations

import chess

# Standard material weights (pawns). Used by material balance, hanging/attack
# pressure, pin burden, and fork qualification in fen_metrics / move_analysis.
PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

# Scales which hanging pieces count in fen_hanging_value. Majors (R/Q) are
# ignored (0.0); pawns and minors count fully (1.0).
HANGING_VULNERABILITY_WEIGHT: dict[chess.PieceType, float] = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 1.0,
    chess.BISHOP: 1.0,
    chess.ROOK: 0.0,
    chess.QUEEN: 0.0,
}

# File/diagonal openness scores for fen_vertical_openness / fen_diagonal_openness
# (and king-file openness inside fen_king_safety): open = no pawns, semi = one side.
_LINE_OPEN_WEIGHT = 1.0
_LINE_SEMI_OPEN_WEIGHT = 0.5

# Pressure contributed by each piece type attacking the king zone in fen_king_safety.
ATTACKER_WEIGHT: dict[chess.PieceType, float] = {
    chess.PAWN: 0.5,
    chess.KNIGHT: 1.0,
    chess.BISHOP: 1.0,
    chess.ROOK: 1.5,
    chess.QUEEN: 2.0,
}
