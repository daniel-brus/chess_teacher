"""E22 candidate loss packing + soft/SF-mix smoke."""

from __future__ import annotations

import numpy as np

from chess_teacher.pipelines.neural_network.candidate_eval import MAX_CANDIDATES, MOVE_FEAT_DIM
from chess_teacher.pipelines.neural_network.candidate_losses import (
    DEFAULT_LOSS_KIND,
    DEFAULT_SF_MIX_ALPHA,
    pack_candidate_targets_for_loss,
    resolve_candidate_loss,
    sf_best_indices_from_eval,
    soft_labels_from_delta_vs_best,
)
from chess_teacher.pipelines.neural_network.material_regime import (
    material_regime_flags,
    only_kings_and_pawns,
    piece_count,
)


def _toy_feats_mask_labels() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3 rows, 4 live candidates; best SF = slot 1; user label = 2 or 1."""
    n, mx, f = 3, MAX_CANDIDATES, MOVE_FEAT_DIM
    feats = np.zeros((n, mx, f), dtype=np.float32)
    mask = np.zeros((n, mx), dtype=np.float32)
    mask[:, :4] = 1.0
    # Packed tanh(delta/5): best=0, others negative.
    delta_i = 2  # delta_vs_best in CANDIDATE_MOVE_FEAT_KEYS
    eval_i = 0  # evaluation_after_user_pov
    for row in range(n):
        for c in range(4):
            # eval: slot1 best
            raw_eval = 1.0 if c == 1 else (0.5 if c == 0 else -0.5)
            feats[row, c, eval_i] = np.tanh(raw_eval / 5.0)
            raw_delta = raw_eval - 1.0
            feats[row, c, delta_i] = np.tanh(raw_delta / 5.0)
    labels = np.asarray([2, 1, 0], dtype=np.int32)
    return feats, mask, labels


def test_soft_labels_peak_on_sf_best() -> None:
    feats, mask, _labels = _toy_feats_mask_labels()
    soft = soft_labels_from_delta_vs_best(feats, mask, temperature_pawns=0.5)
    assert soft.shape == mask.shape
    assert np.allclose(soft.sum(axis=1), 1.0, atol=1e-5)
    assert int(np.argmax(soft[0])) == 1


def test_sf_best_indices() -> None:
    feats, mask, _labels = _toy_feats_mask_labels()
    idx = sf_best_indices_from_eval(feats, mask)
    assert idx.tolist() == [1, 1, 1]


def test_pack_layouts() -> None:
    feats, mask, labels = _toy_feats_mask_labels()
    y_sp = pack_candidate_targets_for_loss(
        loss_kind="sparse", labels=labels, mask=mask, move_feats=feats
    )
    assert y_sp.shape == (3, MAX_CANDIDATES + 1)
    y_soft = pack_candidate_targets_for_loss(
        loss_kind="soft", labels=labels, mask=mask, move_feats=feats
    )
    assert y_soft.shape == (3, 2 * MAX_CANDIDATES + 1)
    y_mix = pack_candidate_targets_for_loss(
        loss_kind="sf_mix", labels=labels, mask=mask, move_feats=feats
    )
    assert y_mix.shape == (3, MAX_CANDIDATES + 2)
    assert y_mix[0, -1] == labels[0]
    assert y_mix[0, MAX_CANDIDATES] == 1  # sf best


def test_default_loss_is_sf_mix_user_only() -> None:
    """Default path = sf_mix packing + α=0 (user CE only)."""
    assert DEFAULT_LOSS_KIND == "sf_mix"
    assert DEFAULT_SF_MIX_ALPHA == 0.0
    feats, mask, labels = _toy_feats_mask_labels()
    y_default = pack_candidate_targets_for_loss(
        loss_kind=DEFAULT_LOSS_KIND, labels=labels, mask=mask, move_feats=feats
    )
    assert y_default.shape == (3, MAX_CANDIDATES + 2)
    loss_fn = resolve_candidate_loss()
    assert loss_fn.__name__ == "masked_candidate_sf_mix_ce"


def test_material_regime_kings_pawns() -> None:
    fen = "8/8/8/4k3/8/3K4/4P3/8 w - - 0 1"
    assert only_kings_and_pawns(fen)
    assert piece_count(fen) == 3
    flags = material_regime_flags(fen)
    assert flags["only_kings_and_pawns"] is True
    assert flags["pieces_le_6"] is True
