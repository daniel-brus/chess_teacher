"""Neural baseline bot input-feed helpers."""

from __future__ import annotations

from types import SimpleNamespace

from chess_teacher.bots.neural_baseline_bot import (
    _model_input_names,
    model_expects_board_input,
)


def test_model_input_names_from_list() -> None:
    model = SimpleNamespace(
        inputs=[
            SimpleNamespace(name="board"),
            SimpleNamespace(name="state:0"),
            SimpleNamespace(name="move_feats"),
        ]
    )
    assert _model_input_names(model) == {"board", "state", "move_feats"}


def test_model_input_names_from_dict() -> None:
    model = SimpleNamespace(inputs={"board": object(), "state": object()})
    assert _model_input_names(model) == {"board", "state"}


def test_model_expects_board_when_three_named_inputs() -> None:
    model = SimpleNamespace(
        inputs=[
            SimpleNamespace(name="board", shape=(None, 8, 8, 17)),
            SimpleNamespace(name="state", shape=(None, 20)),
            SimpleNamespace(name="move_feats", shape=(None, 128, 55)),
        ],
        # Force shape-compat path off; name / count path still True.
        output_shape=(None, 128),
    )
    assert model_expects_board_input(model) is True


def test_model_expects_board_false_for_mlp_two_inputs() -> None:
    model = SimpleNamespace(
        inputs=[
            SimpleNamespace(name="state", shape=(None, 20)),
            SimpleNamespace(name="move_feats", shape=(None, 128, 55)),
        ],
        output_shape=(None, 128),
    )
    assert model_expects_board_input(model) is False
