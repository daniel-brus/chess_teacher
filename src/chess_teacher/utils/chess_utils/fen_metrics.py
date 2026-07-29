from __future__ import annotations

import chess

from chess_teacher.utils.chess_utils.constants import (
    _LINE_OPEN_WEIGHT,
    _LINE_SEMI_OPEN_WEIGHT,
    ATTACKER_WEIGHT,
    HANGING_VULNERABILITY_WEIGHT,
    PIECE_VALUES,
)


def material_balance_white_pov(fen: str) -> float:
    """Return white material minus black material using standard piece values."""
    board = chess.Board(fen)
    white = 0
    black = 0
    for piece in board.piece_map().values():
        value = PIECE_VALUES.get(piece.piece_type, 0)
        if piece.color == chess.WHITE:
            white += value
        else:
            black += value
    return float(white - black)


def _board_from_fen(fen: str) -> chess.Board:
    return chess.Board(fen)


def _line_pawn_openness(board: chess.Board, squares: list[chess.Square]) -> float:
    """Return openness contribution for a file or diagonal (0 if blocked by both pawn colors)."""
    white_pawn = False
    black_pawn = False
    for square in squares:
        piece = board.piece_at(square)
        if piece is None or piece.piece_type != chess.PAWN:
            continue
        if piece.color == chess.WHITE:
            white_pawn = True
        else:
            black_pawn = True
    if not white_pawn and not black_pawn:
        return 1.0
    if white_pawn ^ black_pawn:
        return 0.5
    return 0.0


def _squares_on_sum_diagonal(diagonal_sum: int) -> list[chess.Square]:
    squares: list[chess.Square] = []
    for file_index in range(8):
        rank_index = diagonal_sum - file_index
        if 0 <= rank_index < 8:
            squares.append(chess.square(file_index, rank_index))
    return squares


def _squares_on_diff_diagonal(diagonal_diff: int) -> list[chess.Square]:
    squares: list[chess.Square] = []
    for file_index in range(8):
        rank_index = file_index - diagonal_diff
        if 0 <= rank_index < 8:
            squares.append(chess.square(file_index, rank_index))
    return squares


def _weighted_line_openness(board: chess.Board, squares: list[chess.Square]) -> float:
    line_score = _line_pawn_openness(board, squares)
    if line_score == 1.0:
        return _LINE_OPEN_WEIGHT
    if line_score == 0.5:
        return _LINE_SEMI_OPEN_WEIGHT
    return 0.0


def fen_vertical_openness(fen: str) -> float:
    """Weighted file openness: open file = 1.0, semi-open = 0.5 (sum over 8 files, 0-8)."""
    board = _board_from_fen(fen)
    score = 0.0
    for file_index in range(8):
        file_squares = [chess.square(file_index, rank_index) for rank_index in range(8)]
        score += _weighted_line_openness(board, file_squares)
    return score


def fen_diagonal_openness(fen: str) -> float:
    """Weighted diagonal openness for six center diagonals (three \\ and three /, 0-6)."""
    board = _board_from_fen(fen)
    score = 0.0
    for diagonal_diff in (-1, 0, 1):
        score += _weighted_line_openness(board, _squares_on_diff_diagonal(diagonal_diff))
    for diagonal_sum in (6, 7, 8):
        score += _weighted_line_openness(board, _squares_on_sum_diagonal(diagonal_sum))
    return score


def _pawn_attack_squares(square: chess.Square, color: chess.Color) -> set[chess.Square]:
    file_index = chess.square_file(square)
    rank_index = chess.square_rank(square)
    attacks: set[chess.Square] = set()
    for delta_file in (-1, 1):
        next_file = file_index + delta_file
        if not 0 <= next_file < 8:
            continue
        next_rank = rank_index + (1 if color == chess.WHITE else -1)
        if 0 <= next_rank < 8:
            attacks.add(chess.square(next_file, next_rank))
    return attacks


def fen_pawn_tension(fen: str) -> float:
    """Count pawn pairs on adjacent files that attack each other's square."""
    board = _board_from_fen(fen)
    white_pawns: list[chess.Square] = []
    black_pawns: list[chess.Square] = []
    for square, piece in board.piece_map().items():
        if piece.piece_type != chess.PAWN:
            continue
        if piece.color == chess.WHITE:
            white_pawns.append(square)
        else:
            black_pawns.append(square)

    tension_pairs = 0
    for white_square in white_pawns:
        white_attacks = _pawn_attack_squares(white_square, chess.WHITE)
        for black_square in black_pawns:
            if abs(chess.square_file(white_square) - chess.square_file(black_square)) != 1:
                continue
            black_attacks = _pawn_attack_squares(black_square, chess.BLACK)
            if black_square in white_attacks or white_square in black_attacks:
                tension_pairs += 1
    return float(tension_pairs)


def fen_legal_moves(fen: str) -> tuple[float, float]:
    """Legal move counts for white and black (side-to-move in FEN is ignored)."""
    board = _board_from_fen(fen)
    board.turn = chess.WHITE
    white_moves = float(len(list(board.legal_moves)))
    board.turn = chess.BLACK
    black_moves = float(len(list(board.legal_moves)))
    return white_moves, black_moves


def _king_zone_squares(king_square: chess.Square) -> list[chess.Square]:
    file_index = chess.square_file(king_square)
    rank_index = chess.square_rank(king_square)
    squares: list[chess.Square] = []
    for delta_file in (-1, 0, 1):
        for delta_rank in (-1, 0, 1):
            next_file = file_index + delta_file
            next_rank = rank_index + delta_rank
            if 0 <= next_file < 8 and 0 <= next_rank < 8:
                squares.append(chess.square(next_file, next_rank))
    return squares


def _file_openness_for_king(board: chess.Board, king_square: chess.Square) -> float:
    king_file = chess.square_file(king_square)
    penalty = 0.0
    for file_index in (king_file - 1, king_file, king_file + 1):
        if file_index < 0 or file_index > 7:
            continue
        file_squares = [chess.square(file_index, rank_index) for rank_index in range(8)]
        line_score = _line_pawn_openness(board, file_squares)
        if line_score == 1.0:
            penalty += 1.0
        elif line_score == 0.5:
            penalty += 0.5
    return penalty


def _pawn_shield_squares(king_square: chess.Square, color: chess.Color) -> list[chess.Square]:
    king_file = chess.square_file(king_square)
    king_rank = chess.square_rank(king_square)
    squares: list[chess.Square] = []
    for delta_file in (-1, 0, 1):
        file_index = king_file + delta_file
        if not 0 <= file_index < 8:
            continue
        if color == chess.WHITE:
            rank_range = range(king_rank + 1, min(king_rank + 3, 8))
        else:
            rank_range = range(max(king_rank - 2, 0), king_rank)
        for rank_index in rank_range:
            squares.append(chess.square(file_index, rank_index))
    return squares


def _pawn_shield_bonus(board: chess.Board, king_square: chess.Square, color: chess.Color) -> float:
    bonus = 0.0
    for square in _pawn_shield_squares(king_square, color):
        piece = board.piece_at(square)
        if piece is not None and piece.color == color and piece.piece_type == chess.PAWN:
            bonus += 0.75
    return bonus


def fen_king_safety(board: chess.Board, color: chess.Color) -> float:
    """Higher score means a safer king for ``color``."""
    king_square = board.king(color)
    if king_square is None:
        return 0.0

    enemy_color = not color
    attacker_pressure = 0.0
    for zone_square in _king_zone_squares(king_square):
        for attacker in board.attackers(enemy_color, zone_square):
            attacker_piece = board.piece_at(attacker)
            if attacker_piece is None:
                continue
            attacker_pressure += ATTACKER_WEIGHT.get(attacker_piece.piece_type, 1.0)

    open_file_penalty = _file_openness_for_king(board, king_square)
    shield_bonus = _pawn_shield_bonus(board, king_square, color)
    king_attackers = len(board.attackers(enemy_color, king_square))
    king_attack_penalty = king_attackers * 1.25

    score = 6.0 + shield_bonus - attacker_pressure * 0.12 - open_file_penalty - king_attack_penalty
    return max(0.0, score)


def fen_mean_rank(board: chess.Board, color: chess.Color) -> float:
    """Mean piece rank normalized 0-1 from ``color``'s perspective (higher = more advanced)."""
    ranks: list[float] = []
    for square, piece in board.piece_map().items():
        if piece.color != color:
            continue
        rank_index = chess.square_rank(square)
        ranks.append(rank_index / 7.0 if color == chess.WHITE else (7 - rank_index) / 7.0)
    if not ranks:
        return 0.0
    return sum(ranks) / len(ranks)


def fen_attack_pressure(board: chess.Board, color: chess.Color) -> float:
    """Pressure on ``color``'s pieces from enemy attackers exceeding defenders."""
    enemy_color = not color
    pressure = 0.0
    for square, piece in board.piece_map().items():
        if piece.color != color or piece.piece_type == chess.KING:
            continue
        attackers = len(board.attackers(enemy_color, square))
        defenders = len(board.attackers(color, square))
        if attackers > defenders:
            piece_value = PIECE_VALUES.get(piece.piece_type, 0)
            pressure += piece_value * (attackers - defenders)
    return float(pressure)


def fen_hanging_value(board: chess.Board, color: chess.Color) -> float:
    """Vulnerability-weighted value of ``color``'s hanging pieces."""
    enemy_color = not color
    hanging = 0.0
    for square, piece in board.piece_map().items():
        if piece.color != color or piece.piece_type == chess.KING:
            continue
        attackers = len(board.attackers(enemy_color, square))
        defenders = len(board.attackers(color, square))
        if attackers <= defenders:
            continue
        piece_value = PIECE_VALUES.get(piece.piece_type, 0)
        weight = HANGING_VULNERABILITY_WEIGHT.get(piece.piece_type, 1.0)
        if weight == 0.0:
            continue
        hanging += piece_value * weight * (attackers - defenders)
    return hanging


def _slider_attacks_square(
    board: chess.Board,
    attacker_square: chess.Square,
    target_square: chess.Square,
) -> bool:
    attacker_piece = board.piece_at(attacker_square)
    if attacker_piece is None:
        return False
    if attacker_square == target_square:
        return False
    if not chess.ray(attacker_square, target_square):
        return False
    piece_type = attacker_piece.piece_type
    if piece_type == chess.BISHOP:
        return bool(chess.BB_DIAG_MASKS[attacker_square] & chess.BB_SQUARES[target_square])
    if piece_type == chess.ROOK:
        return bool(
            (chess.BB_RANK_MASKS[attacker_square] | chess.BB_FILE_MASKS[attacker_square])
            & chess.BB_SQUARES[target_square]
        )
    if piece_type == chess.QUEEN:
        return True
    return False


def _pin_line_target(
    board: chess.Board,
    color: chess.Color,
    pinned_square: chess.Square,
    attacker_square: chess.Square,
) -> tuple[chess.Square, chess.Piece] | None:
    """Return the friendly piece behind ``pinned_square`` on the pin ray."""
    ray = chess.ray(attacker_square, pinned_square)
    if not ray:
        return None
    beyond = ray & ~chess.BB_SQUARES[pinned_square] & ~chess.BB_SQUARES[attacker_square]
    closest_target: tuple[int, chess.Square, chess.Piece] | None = None
    for target_square in chess.SquareSet(beyond):
        target_piece = board.piece_at(target_square)
        if target_piece is None or target_piece.color != color:
            continue
        distance = chess.square_distance(pinned_square, target_square)
        if closest_target is None or distance < closest_target[0]:
            closest_target = (distance, target_square, target_piece)
    if closest_target is None:
        return None
    return closest_target[1], closest_target[2]


def _pinning_line(
    board: chess.Board,
    color: chess.Color,
    pinned_square: chess.Square,
) -> tuple[chess.Square, chess.Square, chess.Piece] | None:
    """Return (attacker square, target square, target piece) for a pin on ``pinned_square``."""
    if not board.is_pinned(color, pinned_square):
        return None
    enemy_color = not color
    for attacker_square in board.attackers(enemy_color, pinned_square):
        if not _slider_attacks_square(board, attacker_square, pinned_square):
            continue
        target = _pin_line_target(board, color, pinned_square, attacker_square)
        if target is not None:
            target_square, target_piece = target
            return attacker_square, target_square, target_piece
    return None


def _pin_burden_for_piece(
    board: chess.Board,
    color: chess.Color,
    pinned_square: chess.Square,
    pinned_piece: chess.Piece,
) -> float:
    pin_line = _pinning_line(board, color, pinned_square)
    if pin_line is None:
        return PIECE_VALUES.get(pinned_piece.piece_type, 0) * 0.5

    attacker_square, _target_square, target_piece = pin_line
    pinned_value = PIECE_VALUES.get(pinned_piece.piece_type, 0)
    target_value = PIECE_VALUES.get(target_piece.piece_type, 0)

    if target_piece.piece_type == chess.KING:
        burden = float(pinned_value)
    elif pinned_value > target_value:
        burden = float(pinned_value - target_value)
    else:
        burden = pinned_value * 0.35

    attacker_piece = board.piece_at(attacker_square)
    if attacker_piece is not None:
        attacker_defenders = len(board.attackers(attacker_piece.color, attacker_square))
        if attacker_defenders == 0:
            burden *= 1.25

    return burden


def fen_pin_value(board: chess.Board, color: chess.Color) -> float:
    """Pin burden on ``color``'s pieces (higher = worse for that side)."""
    pin_burden = 0.0
    for square, piece in board.piece_map().items():
        if piece.color != color or piece.piece_type == chess.KING:
            continue
        if not board.is_pinned(color, square):
            continue
        pin_burden += _pin_burden_for_piece(board, color, square, piece)
    return float(pin_burden)


def _attacks_by_piece(board: chess.Board, square: chess.Square) -> set[chess.Square]:
    if board.piece_at(square) is None:
        return set()
    return set(board.attacks(square))


def fen_is_in_check(fen: str) -> bool:
    """True when the side to move in ``fen`` is in check."""
    board = _board_from_fen(fen)
    return board.is_check()


def fen_has_hanging_piece(board: chess.Board, color: chess.Color) -> bool:
    """True when ``color`` has at least one hanging pawn or minor piece."""
    return fen_hanging_value(board, color) > 0.0


def fen_has_castling_rights(board: chess.Board, color: chess.Color) -> bool:
    return board.has_castling_rights(color)


def fen_game_phase(fen: str) -> tuple[bool, bool, bool]:
    """
    Return ``(is_opening, is_middle_game, is_end_game)`` from a FEN snapshot.

    Mutually exclusive buckets based on piece count, queens, and fullmove number.
    """
    board = _board_from_fen(fen)
    piece_count = len(board.piece_map())
    queen_count = sum(1 for piece in board.piece_map().values() if piece.piece_type == chess.QUEEN)
    is_end_game = piece_count <= 12 or (queen_count == 0 and piece_count <= 14)
    is_opening = not is_end_game and board.fullmove_number <= 12
    is_middle_game = not is_opening and not is_end_game
    return is_opening, is_middle_game, is_end_game
