"""Unit tests for update-rule AB helpers."""

from __future__ import annotations

from chess_teacher.pipelines.neural_network.offline_update_rule_ab import (
    mix_replay,
    split_train_by_game,
)
from chess_teacher.utils.chess_utils import Color


class _D:
    def __init__(self, game_id: str, ply: int) -> None:
        self.game_id = game_id
        self.ply = ply
        self.color = Color.WHITE


def test_split_train_by_game_keeps_games_intact() -> None:
    # Need >=8 moves each side (split guard); four games x 5 moves.
    rows = (
        [_D("g1", i) for i in range(5)]
        + [_D("g2", i) for i in range(5)]
        + [_D("g3", i) for i in range(5)]
        + [_D("g4", i) for i in range(5)]
    )
    parent, finetune = split_train_by_game(rows, parent_frac=0.5, seed=0)  # type: ignore[arg-type]
    parent_games = {d.game_id for d in parent}
    finetune_games = {d.game_id for d in finetune}
    assert parent_games.isdisjoint(finetune_games)
    assert parent_games | finetune_games == {"g1", "g2", "g3", "g4"}
    assert len(parent) >= 8 and len(finetune) >= 8


def test_mix_replay_one_to_one_cap() -> None:
    old = [_D("old", i) for i in range(10)]
    new = [_D("new", i) for i in range(4)]
    mixed = mix_replay(old, new, seed=1)  # type: ignore[arg-type]
    assert len(mixed) == 8  # 4 old + 4 new
    assert sum(1 for d in mixed if d.game_id == "new") == 4
