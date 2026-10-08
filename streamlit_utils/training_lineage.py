"""Trunk-and-branch markup for baseline and personal models."""

from __future__ import annotations

import html

import streamlit as st

from chess_teacher.pipelines.neural_network.training_progress import LineageLayout, LineageNode
from streamlit_utils.layout import ingest_css

_CSS_SESSION_KEY = "_ct_training_lineage_css"
_LINEAGE_CSS = """
<style>
.ct-lineage-trunk {
    border-left: 2px solid color-mix(in srgb, var(--text-color) 35%, transparent);
    margin: 0.25rem 0 0.25rem 0.35rem;
    padding-left: 0.9rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}
.ct-lineage-row, .ct-lineage-branch {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.45rem;
}
.ct-lineage-node {
    background: var(--secondary-background-color);
    border-radius: 0.5rem;
    padding: 0.35rem 0.65rem;
    min-width: 5.5rem;
    box-shadow: inset 3px 0 0 var(--text-color);
}
.ct-lineage-node.promoted { box-shadow: inset 3px 0 0 #2e7d32; }
.ct-lineage-node.candidate { box-shadow: inset 3px 0 0 #ef6c00; }
.ct-lineage-node.replaced { box-shadow: inset 3px 0 0 #90a4ae; }
.ct-lineage-title { font-weight: 600; }
.ct-lineage-meta { font-size: 0.8rem; opacity: 0.8; }
.ct-lineage-join { opacity: 0.55; }
.ct-lineage-unattached { margin-top: 0.9rem; }
.ct-lineage-heading {
    font-size: 0.9rem;
    opacity: 0.8;
    margin-bottom: 0.35rem;
}
</style>
"""


def lineage_markup(layout: LineageLayout) -> str:
    """HTML for the trunk. Empty when there is nothing to draw."""
    if not layout.trunk and not layout.unattached:
        return ""
    parts: list[str] = []
    if layout.trunk:
        rows = "".join(_trunk_row(row.baseline, row.branch) for row in layout.trunk)
        parts.append(f'<div class="ct-lineage-trunk">{rows}</div>')
    if layout.unattached:
        nodes = _chain(layout.unattached)
        parts.append(
            '<div class="ct-lineage-unattached">'
            '<div class="ct-lineage-heading">Your models with no baseline parent</div>'
            f'<div class="ct-lineage-branch">{nodes}</div>'
            "</div>"
        )
    return "".join(parts)


def render_lineage(layout: LineageLayout) -> None:
    markup = lineage_markup(layout)
    if not markup:
        st.info("No models to show.")
        return
    if not st.session_state.get(_CSS_SESSION_KEY):
        st.session_state[_CSS_SESSION_KEY] = True
        ingest_css(_LINEAGE_CSS)
    st.html(markup)


def _trunk_row(baseline: LineageNode, branch: tuple[LineageNode, ...]) -> str:
    branch_html = f'<div class="ct-lineage-branch">{_chain(branch)}</div>' if branch else ""
    return f'<div class="ct-lineage-row">{_node(baseline)}{branch_html}</div>'


def _chain(nodes: tuple[LineageNode, ...]) -> str:
    pieces: list[str] = []
    for index, node in enumerate(nodes):
        if index:
            pieces.append('<span class="ct-lineage-join">→</span>')
        pieces.append(_node(node))
    return "".join(pieces)


def _node(node: LineageNode) -> str:
    status = html.escape(node.status_key, quote=True)
    title = html.escape(node.title)
    meta = html.escape(node.meta)
    return (
        f'<div class="ct-lineage-node {status}">'
        f'<div class="ct-lineage-title">{title}</div>'
        f'<div class="ct-lineage-meta">{meta}</div>'
        "</div>"
    )
