from chess_teacher.pipelines.preprocessing.chess_com_openings import chess_com_opening_slug_from_pgn
from chess_teacher.pipelines.preprocessing.transformations import CleanPGNTransformation

CHESS_COM_SAMPLE = (
    '[Event "Live Chess"]\n[Site "Chess.com"]\n[Result "1-0"]\n\n'
    "1. e4 {[%clk 0:09:49.9]} 1... e5 {[%clk 0:09:58.5]} 2. Nf3 {[%clk 0:09:57.9]} "
    "2... Nc6 {[%clk 0:09:48.4]} 3. d3 1-0\n"
)

LICHESS_SAMPLE = (
    '[Event "rated blitz game"]\n[Site "https://lichess.org/AsJuVGA6"]\n[Result "1-0"]\n\n'
    "1. e4 e5 2. Nc3 f6 3. f4 Ne7 4. fxe5 fxe5 5. Nf3 1-0\n"
)

NO_MOVES_SAMPLE = '[Event "Live Chess"]\n[Site "Chess.com"]\n[Result "1-0"]\n\n1-0\n'


def test_clean_pgn_chess_com_strips_headers_clocks_and_result() -> None:
    assert CleanPGNTransformation._clean_pgn(CHESS_COM_SAMPLE) == "1. e4 e5 2. Nf3 Nc6 3. d3"


def test_clean_pgn_lichess_strips_headers_and_result() -> None:
    assert (
        CleanPGNTransformation._clean_pgn(LICHESS_SAMPLE)
        == "1. e4 e5 2. Nc3 f6 3. f4 Ne7 4. fxe5 fxe5 5. Nf3"
    )


def test_clean_pgn_no_moves_returns_empty() -> None:
    assert CleanPGNTransformation._clean_pgn(NO_MOVES_SAMPLE) == ""


def test_clean_pgn_none_or_blank_returns_empty() -> None:
    assert CleanPGNTransformation._clean_pgn(None) == ""
    assert CleanPGNTransformation._clean_pgn("   ") == ""


def test_chess_com_opening_slug_from_pgn() -> None:
    pgn = '[ECO "A48"]\n[ECOUrl "https://www.chess.com/openings/Torre-Attack-Fianchetto-Defense"]\n'
    assert chess_com_opening_slug_from_pgn(pgn) == "Torre-Attack-Fianchetto-Defense"
