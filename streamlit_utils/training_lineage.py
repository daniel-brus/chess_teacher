"""Training lineage as a Mermaid dual-track flowchart (``st.mermaid_chart``)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import streamlit as st

from chess_teacher.pipelines.neural_network.training_progress import LineageLayout, LineageNode

# Without a nuclear cutoff local DB can hold 100+ archived baselines.
_DEFAULT_MAX_EVENTS = 36


@dataclass(frozen=True)
class _Placed:
    kind: str  # baseline | personal
    node: LineageNode
    parent_baseline: str | None


def lineage_mermaid(
    layout: LineageLayout,
    *,
    max_events: int = _DEFAULT_MAX_EVENTS,
) -> str:
    """Mermaid flowchart: Baseline left, Yours right, chronological within each track."""
    if not layout.trunk and not layout.unattached:
        return ""
    placed, total = _place_chronological(layout, max_events=max_events)
    if not placed:
        return ""

    baselines = [item for item in placed if item.kind == "baseline"]
    personals = [item for item in placed if item.kind == "personal"]
    baseline_ids = {item.node.version: _node_id("b", item.node.version) for item in baselines}

    lines: list[str] = [
        "flowchart LR",
        "  classDef promoted fill:#2e7d32,stroke:#1b5e20,stroke-width:2px,color:#fff",
        "  classDef parent fill:#fff8e1,stroke:#f9a825,stroke-width:3px,color:#1a1a1a",
        "  classDef promotedParent fill:#2e7d32,stroke:#f9a825,stroke-width:4px,color:#fff",
        "  classDef candidate fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,color:#1a1a1a",
        "  classDef replaced fill:#eceff1,stroke:#90a4ae,stroke-width:1px,color:#455a64",
        "  classDef yours fill:#e8eaf6,stroke:#5c6bc0,stroke-width:2px,color:#1a1a1a",
    ]

    if baselines:
        lines.append('  subgraph baseline_track ["Baseline"]')
        lines.append("    direction LR")
        for item in baselines:
            nid = baseline_ids[item.node.version]
            lines.append(f"    {nid}[{_node_label(item.node)}]")
        for left, right in pairwise(baselines):
            lines.append(
                f"    {baseline_ids[left.node.version]} --> {baseline_ids[right.node.version]}"
            )
        lines.append("  end")

    personal_ids: dict[str, str] = {}
    if personals:
        lines.append('  subgraph yours_track ["Yours"]')
        lines.append("    direction LR")
        for index, item in enumerate(personals):
            # Personal versions can collide with baseline versions.
            nid = _node_id("p", f"{item.node.version}_{index}")
            personal_ids[f"{item.node.version}:{index}"] = nid
            lines.append(f"    {nid}[{_node_label(item.node)}]")
        for left, right in pairwise(personal_ids.values()):
            lines.append(f"    {left} --> {right}")
        lines.append("  end")

    for index, item in enumerate(personals):
        if not item.parent_baseline or item.parent_baseline not in baseline_ids:
            continue
        child = personal_ids[f"{item.node.version}:{index}"]
        parent = baseline_ids[item.parent_baseline]
        lines.append(f"  {parent} -.-> {child}")

    for item in baselines:
        nid = baseline_ids[item.node.version]
        lines.append(f"  class {nid} {_class_name(item.node, yours=False)}")
    for index, item in enumerate(personals):
        nid = personal_ids[f"{item.node.version}:{index}"]
        lines.append(f"  class {nid} {_class_name(item.node, yours=True)}")

    if total > len(placed):
        # Mermaid has no caption; caller shows Streamlit caption from return side-channel.
        pass
    return "\n".join(lines)


def lineage_truncation_note(
    layout: LineageLayout,
    *,
    max_events: int = _DEFAULT_MAX_EVENTS,
) -> str | None:
    placed, total = _place_chronological(layout, max_events=max_events)
    if total <= len(placed):
        return None
    return (
        f"Showing latest {len(placed)} of {total} models (process order). "
        "Stamp a nuclear lineage cutoff to drop old history."
    )


def render_lineage(
    layout: LineageLayout,
    *,
    max_events: int = _DEFAULT_MAX_EVENTS,
) -> None:
    body = lineage_mermaid(layout, max_events=max_events)
    if not body:
        st.info("No models to show.")
        return
    note = lineage_truncation_note(layout, max_events=max_events)
    if note:
        st.caption(note)
    st.mermaid_chart(body, width="stretch")


def _place_chronological(
    layout: LineageLayout,
    *,
    max_events: int,
) -> tuple[tuple[_Placed, ...], int]:
    events: list[tuple[str, LineageNode, str | None]] = []
    for row in layout.trunk:
        events.append(("baseline", row.baseline, None))
        for personal in row.branch:
            events.append(("personal", personal, row.baseline.version))
    for personal in layout.unattached:
        events.append(("personal", personal, None))
    events.sort(key=lambda item: (item[1].trained_at, item[1].version))
    total = len(events)
    if max_events > 0 and total > max_events:
        events = events[-max_events:]
    placed = tuple(
        _Placed(kind=kind, node=node, parent_baseline=parent) for kind, node, parent in events
    )
    return placed, total


def _node_id(prefix: str, version: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in version)
    return f"{prefix}_{safe}"


def _node_label(node: LineageNode) -> str:
    title = _escape_label(node.title)
    meta = _escape_label(_short_meta(node))
    return f'"{title}<br/>{meta}"'


def _escape_label(text: str) -> str:
    return (
        text
        .replace('"', "'")
        .replace("[", "(")
        .replace("]", ")")
        .replace("{", "(")
        .replace("}", ")")
    )


def _short_meta(node: LineageNode) -> str:
    parts: list[str] = []
    if node.status_key == "promoted":
        parts.append("Promoted")
    elif node.status_key == "candidate":
        parts.append("Candidate")
    else:
        parts.append("Replaced")
    if node.is_parent:
        parts.append("Parent")
    if node.parent_version:
        parts.append(f"from {node.parent_version}")
    return " | ".join(parts)


def _class_name(node: LineageNode, *, yours: bool) -> str:
    if node.is_parent and node.status_key == "promoted":
        return "promotedParent"
    if node.is_parent:
        return "parent"
    if node.status_key == "promoted":
        return "promoted"
    if node.status_key == "candidate":
        return "candidate"
    if yours:
        return "yours"
    return "replaced"
