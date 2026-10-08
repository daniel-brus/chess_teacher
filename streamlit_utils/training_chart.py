"""Altair chart for training scores. Accuracy and pawn gap use separate axes."""

from __future__ import annotations

from collections.abc import Sequence

import altair as alt
import polars as pl

from chess_teacher.pipelines.neural_network.training_progress import ChartPoint, ScoreAxis

_CHART_HEIGHT = 420


def build_score_chart(points: Sequence[ChartPoint]) -> alt.Chart | alt.LayerChart | None:
    """Line chart. Accuracy is solid on the left. Pawn gap is dashed on the right."""
    if not points:
        return None
    frame = pl.DataFrame({
        "scored_at": [point.scored_at for point in points],
        "series": [point.series for point in points],
        "value": [point.value for point in points],
        "axis": [point.axis.value for point in points],
        "version": [point.version for point in points],
        "status": [point.status for point in points],
        "n_eval": [point.n_eval for point in points],
    })
    domain = sorted({point.series for point in points})
    accuracy = frame.filter(pl.col("axis") == ScoreAxis.ACCURACY.value)
    pawns = frame.filter(pl.col("axis") == ScoreAxis.PAWNS.value)
    layers: list[alt.Chart] = []
    if accuracy.height > 0:
        layers.append(
            _line_layer(
                accuracy,
                title="Accuracy",
                zero=False,
                dashed=False,
                domain=domain,
                show_legend=True,
            )
        )
    if pawns.height > 0:
        layers.append(
            _line_layer(
                pawns,
                title="Pawns vs Stockfish",
                zero=True,
                dashed=True,
                domain=domain,
                show_legend=accuracy.height == 0,
            )
        )
    if len(layers) == 1:
        return layers[0].properties(height=_CHART_HEIGHT)
    return alt.layer(*layers).resolve_scale(y="independent").properties(height=_CHART_HEIGHT)


def _line_layer(
    frame: pl.DataFrame,
    *,
    title: str,
    zero: bool,
    dashed: bool,
    domain: list[str],
    show_legend: bool,
) -> alt.Chart:
    color = alt.Color(
        "series:N",
        title="Line",
        scale=alt.Scale(domain=domain),
        legend=alt.Legend(orient="bottom", columns=2, labelLimit=240) if show_legend else None,
    )
    if dashed:
        mark = alt.Chart(frame).mark_line(point=True, strokeDash=[5, 3])
    else:
        mark = alt.Chart(frame).mark_line(point=True)
    return mark.encode(
        x=alt.X("scored_at:T", title="When the model was scored"),
        y=alt.Y("value:Q", title=title, scale=alt.Scale(zero=zero)),
        color=color,
        tooltip=[
            alt.Tooltip("series:N", title="Line"),
            alt.Tooltip("version:N", title="Version"),
            alt.Tooltip("status:N", title="Status"),
            alt.Tooltip("n_eval:Q", title="Eval rows", format=","),
            alt.Tooltip("value:Q", title="Value", format=".3f"),
            alt.Tooltip("scored_at:T", title="Scored"),
        ],
    )
