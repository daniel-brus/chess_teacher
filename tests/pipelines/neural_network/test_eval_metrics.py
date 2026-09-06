"""Unit tests for stratified candidate-style eval metrics."""

from __future__ import annotations

import numpy as np

from chess_teacher.pipelines.neural_network.candidate_eval import (
    CANDIDATE_MOVE_FEAT_KEYS,
    MOVE_FEAT_DIM,
)
from chess_teacher.pipelines.neural_network.eval_metrics import (
    EvalMetrics,
    compute_candidate_style_metrics,
    details_from_packed,
    format_error_shortlist,
    format_eval_delta,
    format_phase_eval_rows,
    phase_from_features,
    slice_datums_by_phase,
)


def _synthetic_batch(
    *,
    n: int = 4,
    max_candidates: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Perfect top-1 predictions with mixed SF agree/disagree labels."""
    feats = np.zeros((n, max_candidates, MOVE_FEAT_DIM), dtype=np.float32)
    mask = np.ones((n, max_candidates), dtype=np.float32)
    labels = np.arange(n, dtype=np.int64) % max_candidates
    logits = np.full((n, max_candidates), -10.0, dtype=np.float64)
    for i in range(n):
        logits[i, labels[i]] = 10.0
    delta_i = CANDIDATE_MOVE_FEAT_KEYS.index("delta_vs_best")
    # rows 0,2 agree (delta 0); rows 1,3 disagree
    deltas = [0.0, -1.0, 0.0, -2.0]
    for i, delta in enumerate(deltas[:n]):
        feats[i, labels[i], delta_i] = np.tanh(delta / 5.0)
    plies = [10] * n
    return logits, mask, labels, feats, plies


def test_perfect_predictions_top1_is_one() -> None:
    logits, mask, labels, feats, plies = _synthetic_batch()
    m = compute_candidate_style_metrics(
        logits=logits,
        mask=mask,
        labels=labels,
        move_feats=feats,
        plies=plies,
        n_input=len(labels),
        max_candidates=8,
    )
    assert m.top1_overall == 1.0
    assert m.top3_overall == 1.0
    assert m.top1_sf_agree == 1.0
    assert m.top1_sf_disagree == 1.0
    assert m.n_sf_agree == 2
    assert m.n_sf_disagree == 2


def test_wrong_top1_on_disagree_only() -> None:
    logits, mask, labels, feats, plies = _synthetic_batch(n=4)
    # Flip prediction on disagree rows only (1 and 3)
    for row in (1, 3):
        wrong = (labels[row] + 1) % mask.shape[1]
        logits[row, labels[row]] = -10.0
        logits[row, wrong] = 10.0
    m = compute_candidate_style_metrics(
        logits=logits,
        mask=mask,
        labels=labels,
        move_feats=feats,
        plies=plies,
        n_input=4,
        max_candidates=8,
    )
    assert m.top1_overall == 0.5
    assert m.top1_sf_agree == 1.0
    assert m.top1_sf_disagree == 0.0


def test_as_dict_includes_stratified_keys() -> None:
    logits, mask, labels, feats, plies = _synthetic_batch(n=2)
    m = compute_candidate_style_metrics(
        logits=logits,
        mask=mask,
        labels=labels,
        move_feats=feats,
        plies=plies,
        n_input=2,
        max_candidates=8,
    )
    d = m.as_dict()
    assert "top1_sf_agree" in d
    assert "top1_sf_disagree" in d
    assert d["n_eval"] == 2.0


def _metrics(*, top1: float, agree: float, disagree: float) -> EvalMetrics:
    return EvalMetrics(
        top1_overall=top1,
        top3_overall=0.9,
        top1_sf_agree=agree,
        top3_sf_agree=0.95,
        top1_sf_disagree=disagree,
        top3_sf_disagree=0.4,
        top1_overall_weighted=top1,
        n_eval=100,
        n_dropped=0,
        n_sf_agree=50,
        n_sf_disagree=50,
        sf_disagree_frac=0.5,
    )


def test_format_eval_delta_signs_and_informational_flags() -> None:
    text = format_eval_delta(
        _metrics(top1=0.42, agree=0.70, disagree=0.22),
        _metrics(top1=0.40, agree=0.71, disagree=0.20),
    )
    assert "top1=+0.0200" in text
    assert "agree_t1=-0.0100" in text
    assert "disagree_t1=+0.0200" in text
    assert "informational_beats_top1=true" in text
    assert "informational_beats_disagree=true" in text


def test_phase_from_features_opening_middle_endgame() -> None:
    assert (
        phase_from_features({"is_opening": True, "is_middle_game": False, "is_end_game": False})
        == "opening"
    )
    assert (
        phase_from_features({"is_opening": False, "is_middle_game": True, "is_end_game": False})
        == "middle"
    )
    assert (
        phase_from_features({"is_opening": False, "is_middle_game": False, "is_end_game": True})
        == "endgame"
    )
    assert phase_from_features({}) is None


def test_slice_datums_by_phase_on_synthetic_features() -> None:
    opening = type(
        "D", (), {"features": {"is_opening": True, "is_middle_game": False, "is_end_game": False}}
    )()
    middle = type(
        "D", (), {"features": {"is_opening": False, "is_middle_game": True, "is_end_game": False}}
    )()
    endgame = type(
        "D", (), {"features": {"is_opening": False, "is_middle_game": False, "is_end_game": True}}
    )()
    unknown = type("D", (), {"features": {}})()
    datums = [opening, middle, endgame, unknown]
    assert slice_datums_by_phase(datums, "endgame") == [endgame]  # type: ignore[arg-type]
    assert slice_datums_by_phase(datums, "opening") == [opening]  # type: ignore[arg-type]
    assert slice_datums_by_phase(datums, "middle") == [middle]  # type: ignore[arg-type]


def test_format_phase_eval_rows_marks_empty() -> None:
    text = format_phase_eval_rows({
        "all": _metrics(top1=0.4, agree=0.7, disagree=0.2),
        "endgame": None,
    })
    assert "all top1=0.4000" in text
    assert "endgame (no datums)" in text


def test_details_from_packed_and_error_shortlist() -> None:
    logits, mask, labels, feats, _plies = _synthetic_batch(n=4)
    # Flip top1 on disagree rows 1 and 3.
    for row in (1, 3):
        wrong = (labels[row] + 1) % mask.shape[1]
        logits[row, labels[row]] = -10.0
        logits[row, wrong] = 10.0
    datums = []
    for i, phase_flag in enumerate((
        {"is_opening": True},
        {"is_end_game": True},
        {"is_middle_game": True},
        {"is_end_game": True},
    )):
        datums.append(
            type(
                "D",
                (),
                {
                    "game_id": f"g{i}",
                    "ply": 10 + i,
                    "fen_before": f"fen{i}",
                    "features": phase_flag,
                },
            )()
        )
    details = details_from_packed(
        logits=logits,
        mask=mask,
        labels=labels,
        move_feats=feats,
        kept_datums=datums,  # type: ignore[arg-type]
        max_candidates=8,
    )
    assert details[1].sf_disagree is True
    assert details[1].top1_hit is False
    assert details[1].phase == "endgame"
    text = format_error_shortlist(details, error_limit=10)
    assert "game_id=g1" in text
    assert "fen=fen1" in text
    assert "game_id=g3" in text
    assert "game_id=g0" not in text
