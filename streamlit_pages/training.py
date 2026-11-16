"""Training progress: scores for the baseline and this user's models."""

from __future__ import annotations

import streamlit as st

from chess_teacher.pipelines.neural_network.training_progress import (
    METRIC_OPTIONS,
    SLICE_OPTIONS,
    SUBJECT_OPTIONS,
    WEIGHT_OPTIONS,
    SeriesSelection,
    load_training_progress,
)
from chess_teacher.utils.db.client import get_db_client
from streamlit_utils.login import require_authenticated_user
from streamlit_utils.page_config import configure_page
from streamlit_utils.page_logging import log_page_view
from streamlit_utils.training_viz import format_cutoff_caption, render_training_body


def _labels(options: tuple[tuple[object, str], ...], selected: frozenset[object]) -> list[str]:
    return [label for kind, label in options if kind in selected]


def _kinds(options: tuple[tuple[object, str], ...], labels: list[str]) -> frozenset[object]:
    by_label = {label: kind for kind, label in options}
    return frozenset(by_label[label] for label in labels if label in by_label)


def _selection_from_widgets() -> SeriesSelection:
    defaults = SeriesSelection.default()
    left, right = st.columns(2)
    subject_labels = left.multiselect(
        "Models",
        options=[label for _kind, label in SUBJECT_OPTIONS],
        default=_labels(SUBJECT_OPTIONS, defaults.subjects),
        help="Baseline is the platform trunk. Mine is your personal chain.",
    )
    metric_labels = right.multiselect(
        "Metrics",
        options=[label for _kind, label in METRIC_OPTIONS],
        default=_labels(METRIC_OPTIONS, defaults.metrics),
        help="Turn any combination on. Accuracy lines share an axis. Pawn-gap lines use the other axis.",
    )
    slice_col, weight_col = st.columns(2)
    slice_labels = slice_col.multiselect(
        "Slice",
        options=[label for _kind, label in SLICE_OPTIONS],
        default=_labels(SLICE_OPTIONS, defaults.slices),
        help=(
            "Applies to top-1 and top-3. Agree means the labeled move was Stockfish's best. "
            "The pawn gap always uses the full eval set."
        ),
    )
    weight_labels = weight_col.multiselect(
        "Weighting",
        options=[label for _kind, label in WEIGHT_OPTIONS],
        default=_labels(WEIGHT_OPTIONS, defaults.weights),
        help="Weighted lines use the same sample weights as training.",
    )
    return SeriesSelection(
        subjects=_kinds(SUBJECT_OPTIONS, subject_labels),
        metrics=_kinds(METRIC_OPTIONS, metric_labels),
        slices=_kinds(SLICE_OPTIONS, slice_labels),
        weights=_kinds(WEIGHT_OPTIONS, weight_labels),
    )


configure_page("Training", layout="wide")
user = require_authenticated_user()
log_page_view("Training", user)
db_client = get_db_client()

st.title("Training")
st.caption(
    "Validation scores for the platform baseline and your personal models. "
    "Accuracy is the share of eval moves the model ranked correctly. "
    "The pawn gap is how far the chosen move sits from Stockfish, in pawns. "
    "0 matches the engine."
)

progress = load_training_progress(db_client, user.user_id)
cutoff_note = format_cutoff_caption(progress.model_tree_started_at)
if cutoff_note:
    st.caption(cutoff_note)

if not progress.baselines and not progress.personals:
    st.info("No training models yet.")
    st.stop()

selection = _selection_from_widgets()
render_training_body(progress, selection)
