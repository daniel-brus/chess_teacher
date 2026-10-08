"""Training progress: scores for the baseline and this user's models."""

from __future__ import annotations

import streamlit as st

from chess_teacher.pipelines.neural_network.training_progress import (
    METRIC_OPTIONS,
    SLICE_OPTIONS,
    SUBJECT_OPTIONS,
    WEIGHT_OPTIONS,
    SeriesSelection,
    chart_points,
    lineage_layout,
    load_training_progress,
    score_table_records,
)
from chess_teacher.utils.db.client import get_db_client
from streamlit_utils.login import require_authenticated_user
from streamlit_utils.page_config import configure_page
from streamlit_utils.page_logging import log_page_view
from streamlit_utils.training_chart import build_score_chart
from streamlit_utils.training_lineage import render_lineage


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
if not progress.baselines and not progress.personals:
    st.info("No training models yet.")
    st.stop()

st.subheader("Scores")
selection = _selection_from_widgets()
points = chart_points(progress, selection)
chart = build_score_chart(points)
if chart is None:
    st.info("No scores for the selected lines.")
else:
    has_accuracy = any(point.axis.value == "accuracy" for point in points)
    has_pawns = any(point.axis.value == "pawns" for point in points)
    if has_accuracy and has_pawns:
        st.caption(
            "Solid lines are accuracy, on the left. Dashed lines are the pawn gap, on the right."
        )
    st.altair_chart(chart, width="stretch")

st.subheader("Score table")
records = score_table_records(progress)
if records:
    st.caption(
        "Every stored score for the baseline and your models. The toggles above apply to the graph."
    )
    number = st.column_config.NumberColumn
    column_config: dict[str, object] = {
        "Scored": st.column_config.DatetimeColumn("Scored"),
        "n": number("n", format="%d"),
        "n agree": number("n agree", format="%d"),
        "n disagree": number("n disagree", format="%d"),
    }
    count_columns = {"Scored", "Model", "Status", "Parent baseline", "n", "n agree", "n disagree"}
    for column in records[0]:
        if column not in count_columns:
            column_config[column] = number(column, format="%.3f")
    st.dataframe(records, hide_index=True, width="stretch", column_config=column_config)
else:
    st.info("No validation scores stored yet.")

st.subheader("Lineage")
st.caption(
    "The trunk is the baseline. Your models sit beside the baseline they were trained from. "
    "Promoted is the model currently served. Replaced was served and then swapped out. "
    "Parent is the baseline later personal runs warm-start from."
)
if not progress.personals:
    st.caption("You do not have a personal model yet.")
render_lineage(lineage_layout(progress))
