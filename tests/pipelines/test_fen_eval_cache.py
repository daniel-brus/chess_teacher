from __future__ import annotations

from pathlib import Path

import chess
import pytest

from chess_teacher.pipelines.fen_eval_cache.keys import position_key
from chess_teacher.pipelines.fen_eval_cache.service import (
    FenEvalRequest,
    PositionEvalService,
    reset_position_eval_service_for_tests,
)
from chess_teacher.pipelines.fen_eval_cache.store import MemoryEvalStore

_START = chess.STARTING_FEN
_START_LATER_CLOCKS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 5 10"
_SRC = Path(__file__).resolve().parents[2] / "src" / "chess_teacher"


def test_position_key_strips_clocks() -> None:
    assert position_key(_START) == position_key(_START_LATER_CLOCKS)


def test_hit_skips_recompute() -> None:
    scalar_calls: list[str] = []
    cand_calls: list[str] = []

    def compute_scalar(fen: str) -> float:
        scalar_calls.append(fen)
        return 0.25

    def compute_candidates(fen: str, nodes: int) -> dict[str, float]:
        cand_calls.append(fen)
        return {"e2e4": float(nodes)}

    service = PositionEvalService(
        store=MemoryEvalStore(),
        compute_scalar=compute_scalar,
        compute_candidates=compute_candidates,
        max_ply=32,
        n_workers=1,
    )
    first = service.evaluate(_START, ply=2, eval_depth=12, candidate_nodes=1000)
    second = service.evaluate(_START_LATER_CLOCKS, ply=4, eval_depth=12, candidate_nodes=1000)
    assert first.eval_white_pov == pytest.approx(0.25)
    assert second.eval_white_pov == pytest.approx(0.25)
    assert first.candidate_evals["e2e4"] == pytest.approx(1000)
    assert scalar_calls == [_START]
    assert cand_calls == [_START]


def test_weaker_store_is_upgraded() -> None:
    cand_calls: list[int] = []

    def compute_candidates(_fen: str, nodes: int) -> dict[str, float]:
        cand_calls.append(nodes)
        return {"e2e4": 0.1}

    service = PositionEvalService(
        store=MemoryEvalStore(),
        compute_scalar=lambda _fen: 0.0,
        compute_candidates=compute_candidates,
        max_ply=32,
        n_workers=1,
    )
    service.evaluate(_START, ply=1, eval_depth=12, candidate_nodes=1000)
    upgraded = service.evaluate(_START, ply=1, eval_depth=12, candidate_nodes=50_000)
    assert cand_calls == [1000, 50_000]
    assert upgraded.candidate_nodes == 50_000
    # Stronger row satisfies a later weak ask.
    service.evaluate(_START, ply=1, eval_depth=12, candidate_nodes=1000)
    assert cand_calls == [1000, 50_000]


def test_evaluate_needs_only_fen_and_ply() -> None:
    service = PositionEvalService(
        store=MemoryEvalStore(),
        compute_scalar=lambda _fen: 0.1,
        compute_candidates=lambda _fen, _nodes: {"e2e4": 0.0},
        max_ply=32,
        n_workers=1,
    )
    result = service.evaluate(_START, 8)
    assert result.eval_white_pov == pytest.approx(0.1)
    assert result.candidate_nodes == 50_000


def test_store_if_any_ply_on_the_page_is_early() -> None:
    store = MemoryEvalStore()
    service = PositionEvalService(
        store=store,
        compute_scalar=lambda _fen: 0.3,
        compute_candidates=lambda _fen, _nodes: {"e2e4": 0.1},
        max_ply=32,
        n_workers=1,
    )
    service.evaluate_many([
        FenEvalRequest(_START, ply=80, eval_depth=12, candidate_nodes=1000),
        FenEvalRequest(_START_LATER_CLOCKS, ply=4, eval_depth=12, candidate_nodes=1000),
    ])
    assert position_key(_START) in store.get_many([position_key(_START)])


def test_late_only_page_does_not_insert() -> None:
    store = MemoryEvalStore()
    service = PositionEvalService(
        store=store,
        compute_scalar=lambda _fen: 0.3,
        compute_candidates=lambda _fen, _nodes: {"e2e4": 0.1},
        max_ply=32,
        n_workers=1,
    )
    service.evaluate_many([
        FenEvalRequest(_START, ply=80, eval_depth=12, candidate_nodes=1000),
        FenEvalRequest(_START_LATER_CLOCKS, ply=40, eval_depth=12, candidate_nodes=1000),
    ])
    assert store.get_many([position_key(_START)]) == {}


def test_high_ply_does_not_insert_but_memory_hits() -> None:
    calls = {"n": 0}

    def compute_scalar(_fen: str) -> float:
        calls["n"] += 1
        return 1.0

    store = MemoryEvalStore()
    service = PositionEvalService(
        store=store,
        compute_scalar=compute_scalar,
        compute_candidates=lambda _fen, _nodes: {},
        max_ply=32,
        n_workers=1,
    )
    service.evaluate(_START, ply=80, eval_depth=12, candidate_nodes=1000)
    assert store.get_many([position_key(_START)]) == {}
    service.evaluate(_START, ply=80, eval_depth=12, candidate_nodes=1000)
    assert calls["n"] == 1


def test_lru_evicts_coldest() -> None:
    store = MemoryEvalStore()
    boards = [
        chess.STARTING_FEN,
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    ]
    service = PositionEvalService(
        store=store,
        compute_scalar=lambda fen: float(len(fen)),
        compute_candidates=lambda _fen, _nodes: {},
        max_ply=32,
        capacity=2,
        n_workers=1,
    )
    for fen in boards:
        service.evaluate(fen, ply=1, eval_depth=12, candidate_nodes=100)
    keys = set(store.get_many([position_key(fen) for fen in boards]))
    assert position_key(boards[0]) not in keys
    assert len(keys) == 2


def test_evaluate_many_groups_same_epd() -> None:
    calls: list[str] = []
    service = PositionEvalService(
        store=MemoryEvalStore(),
        compute_scalar=lambda fen: calls.append(fen) or 0.0,
        compute_candidates=lambda _fen, _nodes: {},
        max_ply=32,
        n_workers=1,
    )
    service.evaluate_many([
        FenEvalRequest(_START, ply=2, eval_depth=12, candidate_nodes=100),
        FenEvalRequest(_START_LATER_CLOCKS, ply=8, eval_depth=12, candidate_nodes=100),
    ])
    assert calls == [_START]


def test_production_eval_call_sites_go_through_the_service() -> None:
    allowed = {
        "pipelines/fen_eval_cache/service.py",
        "utils/chess_utils/stockfish.py",
    }
    banned = ("evaluate_white_pov_pawns", "evaluate_all_legal_moves_white_pov")
    offenders: list[str] = []
    for path in _SRC.rglob("*.py"):
        rel = path.relative_to(_SRC).as_posix()
        if rel in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        for name in banned:
            if name in text:
                offenders.append(f"{rel}:{name}")
    assert offenders == []


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    yield
    reset_position_eval_service_for_tests()
