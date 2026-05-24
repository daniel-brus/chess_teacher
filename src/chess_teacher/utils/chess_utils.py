from enum import StrEnum


class Color(StrEnum):
    WHITE = "white"
    BLACK = "black"


class Result(StrEnum):
    WIN = "win"
    DRAW = "draw"
    LOSS = "loss"
    NO_RESULT = "no_result"


class Reason(StrEnum):
    # win/loss reasons:
    CHECKMATE = "checkmate"
    RESIGNATION = "resignation"
    TIMEOUT = "timeout"
    # draw reasons:
    STALEMATE = "stalemate"
    INSUFFICIENT_MATERIAL = "insufficient_material"
    TIMEOUT_INSUFFICIENT_MATERIAL = "timeout_insufficient_material"
    THREEFOLD_REPETITION = "threefold_repetition"
    AGREED_DRAW = "agreed_draw"
    FIFTY_MOVE_RULE = "fifty_move_rule"
    # ambiguous reasons:
    ABANDONED = "abandoned"
    OTHER = "other"
