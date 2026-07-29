from __future__ import annotations

import random
from dataclasses import dataclass

import chess

from chess_teacher.utils.chess_bots import ChessBot, get_bot_preset


@dataclass
class PlayGameState:
    board: chess.Board
    user_color: chess.Color
    preset_key: str
    last_move_uci: str | None
    instance_id: int
    pending_bot_move: bool
    resigned: bool = False


def resolve_user_color(choice: str) -> chess.Color:
    normalized = choice.strip().lower()
    if normalized == "white":
        return chess.WHITE
    if normalized == "black":
        return chess.BLACK
    if normalized == "random":
        return random.choice([chess.WHITE, chess.BLACK])
    raise ValueError(f"Unsupported color choice: {choice!r}")


def user_color_label(color: chess.Color) -> str:
    return "White" if color == chess.WHITE else "Black"


def orientation_for_user(user_color: chess.Color) -> str:
    return "white" if user_color == chess.WHITE else "black"


def is_user_turn(state: PlayGameState) -> bool:
    return (
        not state.resigned
        and state.board.turn == state.user_color
        and not state.board.is_game_over()
    )


def create_bot(preset_key: str) -> ChessBot:
    return get_bot_preset(preset_key).factory()


def close_bot(bot: ChessBot | None) -> None:
    if bot is not None:
        bot.close()


def start_new_game(user_color_choice: str, preset_key: str) -> PlayGameState:
    user_color = resolve_user_color(user_color_choice)
    board = chess.Board()
    return PlayGameState(
        board=board,
        user_color=user_color,
        preset_key=preset_key,
        last_move_uci=None,
        instance_id=0,
        pending_bot_move=user_color == chess.BLACK,
    )


def apply_legal_move(state: PlayGameState, move: chess.Move) -> PlayGameState:
    if move not in state.board.legal_moves:
        raise ValueError(f"Illegal move: {move.uci()}")
    state.board.push(move)
    return PlayGameState(
        board=state.board,
        user_color=state.user_color,
        preset_key=state.preset_key,
        last_move_uci=move.uci(),
        instance_id=state.instance_id + 1,
        pending_bot_move=not state.board.is_game_over() and state.board.turn != state.user_color,
        resigned=state.resigned,
    )


def apply_bot_move(state: PlayGameState, bot: ChessBot) -> PlayGameState:
    if state.board.is_game_over():
        return PlayGameState(
            board=state.board,
            user_color=state.user_color,
            preset_key=state.preset_key,
            last_move_uci=state.last_move_uci,
            instance_id=state.instance_id,
            pending_bot_move=False,
            resigned=state.resigned,
        )
    if state.board.turn == state.user_color:
        return PlayGameState(
            board=state.board,
            user_color=state.user_color,
            preset_key=state.preset_key,
            last_move_uci=state.last_move_uci,
            instance_id=state.instance_id,
            pending_bot_move=False,
            resigned=state.resigned,
        )

    move = bot.choose_move(state.board)
    state.board.push(move)
    return PlayGameState(
        board=state.board,
        user_color=state.user_color,
        preset_key=state.preset_key,
        last_move_uci=move.uci(),
        instance_id=state.instance_id + 1,
        pending_bot_move=False,
        resigned=state.resigned,
    )


def parse_move_uci(board: chess.Board, uci: str) -> chess.Move:
    move = chess.Move.from_uci(uci)
    if move not in board.legal_moves:
        raise ValueError(f"Illegal move from board: {uci!r}")
    return move


def game_status_message(state: PlayGameState) -> str | None:
    if state.resigned:
        return "You resigned. ChessBot wins."

    board = state.board
    user_color = state.user_color
    if not board.is_game_over():
        return None

    outcome = board.outcome()
    if outcome is None:
        return "Game over."

    if outcome.winner is None:
        return f"Draw ({outcome.termination.name.replace('_', ' ').lower()})."

    user_won = outcome.winner == user_color
    termination = outcome.termination.name.replace("_", " ").lower()
    if user_won:
        return f"You win by {termination}!"
    return f"You lose by {termination}."


def move_from_board_event(board: chess.Board, event: dict[str, object]) -> chess.Move | None:
    uci = event.get("uci")
    if not isinstance(uci, str) or not uci:
        return None
    if event.get("kind") == "size":
        return None
    return parse_move_uci(board, uci)
