import polars as pl

from chess_teacher.ingestion.transformations import (
    ApplyChessComOpeningLookupTransformation,
    ApplyLichessOpeningNameTransformation,
    DeriveOpeningTransformation,
)
from chess_teacher.platform.account import AccountPlatform


def test_derive_opening_single_match() -> None:
    df = pl.DataFrame({
        "game_id": ["g1", "g1"],
        "cleaned_pgn": ["1. e4 e5 2. Nf3", "1. e4 e5 2. Nf3"],
        "pgn": ["1. e4", "1. e4 e5"],
        "name": ["King's Pawn Game", "King's Pawn Game: King's Knight Opening"],
        "eco_code_id": ["a", "b"],
    })
    result = DeriveOpeningTransformation().transform(df)
    assert result.height == 1
    assert result["opening_name"][0] == "King's Pawn Game: King's Knight Opening"
    assert result["opening_family"][0] == "King's Pawn Game"


def test_derive_opening_tied_names_concatenated() -> None:
    df = pl.DataFrame({
        "game_id": ["g1", "g1"],
        "cleaned_pgn": ["1. e4 e5", "1. e4 e5"],
        "pgn": ["1. e4 e5", "1. e4 e5"],
        "name": ["B: Variation Two", "A: Variation One"],
        "eco_code_id": ["a", "b"],
    })
    result = DeriveOpeningTransformation().transform(df)
    assert result.height == 1
    assert result["opening_name"][0] == "A: Variation One | B: Variation Two"
    assert result["opening_family"][0] is None


def test_derive_opening_tied_names_shared_family() -> None:
    df = pl.DataFrame({
        "game_id": ["g1", "g1"],
        "cleaned_pgn": ["1. e4 e5", "1. e4 e5"],
        "pgn": ["1. e4 e5", "1. e4 e5"],
        "name": ["Modern Defense: A", "Modern Defense: B"],
        "eco_code_id": ["a", "b"],
    })
    result = DeriveOpeningTransformation().transform(df)
    assert result["opening_name"][0] == "Modern Defense: A | Modern Defense: B"
    assert result["opening_family"][0] == "Modern Defense"


def test_derive_opening_no_match() -> None:
    df = pl.DataFrame({
        "game_id": ["g1"],
        "cleaned_pgn": ["1. d4 d5"],
        "pgn": ["1. e4"],
        "name": ["King's Pawn Game"],
        "eco_code_id": ["a"],
    })
    result = DeriveOpeningTransformation().transform(df)
    assert result["opening_name"][0] is None
    assert result["opening_family"][0] is None


def test_apply_lichess_opening_name_skips_book_match() -> None:
    df = pl.DataFrame({
        "game_id": ["g1", "g1", "g2", "g2"],
        "platform": [AccountPlatform.LICHESS.value] * 4,
        "cleaned_pgn": ["1. e4 e5", "1. e4 e5", "1. d4 d5", "1. d4 d5"],
        "pgn": ["1. e4", "1. e4 e5 2. Nf3", "1. d4", "1. d4 d5"],
        "name": ["King's Pawn Game", "King's Pawn Game: King's Knight Opening"] * 2,
        "eco_code_id": ["a", "b", "c", "d"],
        "opening_name": ["Lichess API: Italian Game", None, None, None],
        "opening_family": ["Lichess API", None, None, None],
    })
    result = DeriveOpeningTransformation().transform(df)
    assert result.height == 2
    lichess_row = result.filter(pl.col("game_id") == "g1").row(0, named=True)
    assert lichess_row["opening_name"] == "Lichess API: Italian Game"
    assert lichess_row["opening_family"] == "Lichess API"
    derived_row = result.filter(pl.col("game_id") == "g2").row(0, named=True)
    assert derived_row["opening_name"] == "King's Pawn Game: King's Knight Opening"


def test_apply_lichess_opening_name_from_platform_field() -> None:
    df = pl.DataFrame({
        "platform": [AccountPlatform.LICHESS.value],
        "platform_opening_name": ["Sicilian Defense: Najdorf Variation"],
    })
    result = ApplyLichessOpeningNameTransformation().transform(df)
    assert result["opening_name"][0] == "Sicilian Defense: Najdorf Variation"
    assert result["opening_family"][0] == "Sicilian Defense"
    assert "platform_opening_name" not in result.columns


def test_apply_chess_com_opening_lookup() -> None:
    lookup = {
        "Torre-Attack-Fianchetto-Defense": "Torre Attack: Fianchetto Defense",
    }
    df = pl.DataFrame({
        "game_id": ["g1", "g2"],
        "chess_com_opening_slug": ["Torre-Attack-Fianchetto-Defense", "unknown-slug"],
        "opening_name": [None, None],
        "opening_family": [None, None],
    })
    result = ApplyChessComOpeningLookupTransformation(lookup=lookup).transform(df)
    assert result["opening_name"][0] == "Torre Attack: Fianchetto Defense"
    assert result["opening_family"][0] == "Torre Attack"
    assert result["opening_name"][1] is None


def test_split_tied_opening_names_round_trip() -> None:
    joined = "Modern Defense: A | Modern Defense: B"
    assert DeriveOpeningTransformation.split_tied_opening_names(joined) == [
        "Modern Defense: A",
        "Modern Defense: B",
    ]
