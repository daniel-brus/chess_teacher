"""Training-page body helpers."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from chess_teacher.pipelines.neural_network.training_progress import (
    ScoreAxis,
    SeriesSelection,
    TrainingProgress,
    chart_points,
    lineage_layout,
    score_table_records,
)
from streamlit_utils.training_chart import build_score_chart
from streamlit_utils.training_lineage import render_lineage


def render_training_body(
    progress: TrainingProgress,
    selection: SeriesSelection,
) -> None:
    """Score chart, lineage diagram, then score table."""
    points = chart_points(progress, selection)
    layout = lineage_layout(progress)

    st.subheader("Scores")
    chart = build_score_chart(points)
    if chart is None:
        st.info("No scores for the selected lines.")
    else:
        has_accuracy = any(point.axis is ScoreAxis.ACCURACY for point in points)
        has_pawns = any(point.axis is ScoreAxis.PAWNS for point in points)
        if has_accuracy and has_pawns:
            st.caption(
                "Solid lines are accuracy, on the left. Dashed lines are the pawn gap, on the right."
            )
        st.altair_chart(chart, width="stretch")

    st.subheader("Lineage")
    st.caption(
        "Two tracks (Baseline | Yours), newest toward the top. "
        "Dotted horizontal = personal trained directly on that baseline. "
        "Solid arrows = weight parent → child."
    )
    if not progress.personals:
        st.caption("You do not have a personal model yet.")
    render_lineage(layout)

    records = score_table_records(progress)
    with st.expander(f"Score table ({len(records)} rows)", expanded=False):
        if not records:
            st.info("No validation scores stored yet.")
        else:
            st.caption(
                "Every stored score for the baseline and your models. "
                "The toggles above apply to the graph."
            )
            number = st.column_config.NumberColumn
            column_config: dict[str, object] = {
                "Scored": st.column_config.DatetimeColumn("Scored"),
                "n": number("n", format="%d"),
                "n agree": number("n agree", format="%d"),
                "n disagree": number("n disagree", format="%d"),
            }
            count_columns = {
                "Scored",
                "Model",
                "Status",
                "Parent baseline",
                "n",
                "n agree",
                "n disagree",
            }
            for column in records[0]:
                if column not in count_columns:
                    column_config[column] = number(column, format="%.3f")
            st.dataframe(records, hide_index=True, width="stretch", column_config=column_config)


def format_cutoff_caption(started_at: datetime | None) -> str | None:
    if started_at is None:
        return None
    return (
        f"Lineage cutoff active since {started_at.isoformat(sep=' ', timespec='minutes')} "
        "(models trained before a nuclear reset are hidden)."
    )
