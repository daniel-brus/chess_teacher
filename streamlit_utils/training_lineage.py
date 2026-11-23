"""Training lineage as a fixed-size vertical weight-parentage diagram (PNG)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from io import BytesIO

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from chess_teacher.pipelines.neural_network.training_progress import LineageLayout, LineageNode

# Without a nuclear cutoff local DB can hold 100+ archived baselines.
_DEFAULT_MAX_EVENTS = 36
_PERSONAL_PARENT_KIND = "personal"
_BASELINE_PARENT_KIND = "baseline"

# Fixed display canvas — content is scaled to fit; image size never changes.
_CANVAS_W = 920
_CANVAS_H = 560
_NODE_W = 160
_NODE_H = 64
_GAP_X = 48
_GAP_Y = 28
_MARGIN = 24
_LABEL_W = 70
_DISPLAY_WIDTH = 920


@dataclass(frozen=True)
class _Placed:
    kind: str  # baseline | personal
    node: LineageNode
    parent_baseline: str | None


@dataclass(frozen=True)
class LineageEdge:
    """Weight-parentage edge used by tests and the renderer."""

    source: str
    target: str
    style: str  # solid | dotted
    source_kind: str
    target_kind: str


@dataclass(frozen=True)
class _Box:
    key: str
    kind: str
    node: LineageNode
    x: float
    y: float
    parent_baseline: str | None = None


def lineage_edges(
    layout: LineageLayout,
    *,
    max_events: int = _DEFAULT_MAX_EVENTS,
) -> tuple[LineageEdge, ...]:
    """Return parent→child edges after the same truncation the diagram uses."""
    placed, _total = _place_chronological(layout, max_events=max_events)
    return _edges_from_placed(placed)


def lineage_truncation_note(
    layout: LineageLayout,
    *,
    max_events: int = _DEFAULT_MAX_EVENTS,
) -> str | None:
    placed, total = _place_chronological(layout, max_events=max_events)
    if total <= len(placed):
        return None
    return (
        f"Showing latest {len(placed)} of {total} models. "
        "Stamp a nuclear lineage cutoff to drop old history."
    )


def render_lineage(
    layout: LineageLayout,
    *,
    max_events: int = _DEFAULT_MAX_EVENTS,
) -> None:
    placed, _total = _place_chronological(layout, max_events=max_events)
    if not placed:
        st.info("No models to show.")
        return
    note = lineage_truncation_note(layout, max_events=max_events)
    if note:
        st.caption(note)
    _render_lineage_legend()
    st.image(_render_png(placed), width=_DISPLAY_WIDTH)


def _render_lineage_legend() -> None:
    """Compact color key — status lives here, not inside every node."""
    chips = [
        (
            "<span style='display:inline-flex;align-items:center;gap:6px;margin-right:14px;'>"
            "<span style='width:18px;height:14px;border-radius:4px;background:#16a34a;"
            "border:1px solid #15803d;display:inline-block;'></span>"
            "<span style='font-size:0.85rem;color:#475569;'>Currently promoted</span></span>"
        ),
        (
            "<span style='display:inline-flex;align-items:center;gap:6px;margin-right:14px;'>"
            "<span style='width:18px;height:14px;border-radius:4px;background:#dcedd6;"
            "border:1px solid #84a27c;display:inline-block;'></span>"
            "<span style='font-size:0.85rem;color:#475569;'>Was promoted</span></span>"
        ),
        (
            "<span style='display:inline-flex;align-items:center;gap:6px;margin-right:14px;'>"
            "<span style='width:18px;height:14px;border-radius:4px;background:#f8fafc;"
            "border:1px solid #94a3b8;display:inline-block;'></span>"
            "<span style='font-size:0.85rem;color:#475569;'>Never promoted</span></span>"
        ),
        (
            "<span style='display:inline-flex;align-items:center;gap:6px;margin-right:14px;'>"
            "<span style='width:18px;height:14px;border-radius:4px;background:#f8fafc;"
            "border:3px solid #d4a017;box-sizing:border-box;display:inline-block;'></span>"
            "<span style='font-size:0.85rem;color:#475569;'>Parent (gold band)</span></span>"
        ),
    ]
    st.html(
        "<div style='display:flex;flex-wrap:wrap;align-items:center;gap:4px 0;margin:0 0 8px 0;'>"
        + "".join(chips)
        + "</div>"
    )


def _render_png(placed: tuple[_Placed, ...]) -> Image.Image:
    boxes = _layout_boxes(placed)
    canvas = Image.new("RGBA", (_CANVAS_W, _CANVAS_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    font = _font(14)
    font_small = _font(11)
    font_label = _font(13)

    if not boxes:
        draw.text((24, 48), "No models to show.", fill=(80, 80, 80, 255), font=font)
        return canvas

    # Logical bounds → scale into fixed canvas with padding.
    min_x = min(box.x for box in boxes.values())
    min_y = min(box.y for box in boxes.values())
    max_x = max(box.x + _NODE_W for box in boxes.values())
    max_y = max(box.y + _NODE_H for box in boxes.values())
    content_w = max(max_x - min_x, 1)
    content_h = max(max_y - min_y, 1)
    pad = 36
    label_space = _LABEL_W + 8
    avail_w = _CANVAS_W - 2 * pad - label_space
    avail_h = _CANVAS_H - 2 * pad
    scale = min(1.0, avail_w / content_w, avail_h / content_h)
    offset_x = pad + label_space + (avail_w - content_w * scale) / 2 - min_x * scale
    offset_y = pad + (avail_h - content_h * scale) / 2 - min_y * scale

    def tx(x: float) -> float:
        return x * scale + offset_x

    def ty(y: float) -> float:
        return y * scale + offset_y

    scaled: dict[str, _Box] = {
        key: _Box(
            key=box.key,
            kind=box.kind,
            node=box.node,
            x=tx(box.x),
            y=ty(box.y),
            parent_baseline=box.parent_baseline,
        )
        for key, box in boxes.items()
    }
    node_w = _NODE_W * scale
    node_h = _NODE_H * scale

    baseline_boxes = [box for box in scaled.values() if box.kind == "baseline"]
    personal_boxes = [box for box in scaled.values() if box.kind == "personal"]
    if baseline_boxes:
        draw.text(
            (10, min(box.y for box in baseline_boxes) - 18),
            "Baseline",
            fill=(69, 90, 100, 255),
            font=font_label,
        )
    if personal_boxes:
        draw.text(
            (min(box.x for box in personal_boxes), min(box.y for box in personal_boxes) - 18),
            "Yours",
            fill=(69, 90, 100, 255),
            font=font_label,
        )

    for edge in _edges_from_placed(placed):
        src = scaled.get(_box_key(edge.source_kind, edge.source))
        dst = scaled.get(_box_key(edge.target_kind, edge.target))
        if src is None or dst is None:
            continue
        _draw_edge(draw, src, dst, style=edge.style, node_w=node_w, node_h=node_h)

    for box in scaled.values():
        _draw_node(
            draw,
            box,
            node_w=node_w,
            node_h=node_h,
            font=font,
            font_small=font_small,
        )

    buffer = BytesIO()
    canvas.save(buffer, format="PNG")
    buffer.seek(0)
    return Image.open(buffer)


def _edges_from_placed(placed: tuple[_Placed, ...]) -> tuple[LineageEdge, ...]:
    baselines = [item for item in placed if item.kind == "baseline"]
    personals = [item for item in placed if item.kind == "personal"]
    baseline_versions = {item.node.version for item in baselines}
    personal_versions = {item.node.version for item in personals}
    edges: list[LineageEdge] = []

    for item in baselines:
        parent = item.node.parent_version
        if parent and parent in baseline_versions:
            edges.append(
                LineageEdge(
                    source=parent,
                    target=item.node.version,
                    style="solid",
                    source_kind="baseline",
                    target_kind="baseline",
                )
            )

    for item in personals:
        if item.node.parent_kind != _PERSONAL_PARENT_KIND:
            continue
        parent = item.node.parent_version
        if parent and parent in personal_versions:
            edges.append(
                LineageEdge(
                    source=parent,
                    target=item.node.version,
                    style="solid",
                    source_kind="personal",
                    target_kind="personal",
                )
            )

    for item in personals:
        if not _is_personal_arm_root(item.node, personal_versions=personal_versions):
            continue
        baseline_key = _baseline_parent_key(item)
        if not baseline_key or baseline_key not in baseline_versions:
            continue
        edges.append(
            LineageEdge(
                source=baseline_key,
                target=item.node.version,
                style="dotted",
                source_kind="baseline",
                target_kind="personal",
            )
        )
    return tuple(edges)


def _layout_boxes(placed: tuple[_Placed, ...]) -> dict[str, _Box]:
    """Two vertical tracks; fork rows linked horizontally; gaps spread evenly.

    - Left column = all baselines (newest toward the top).
    - Right column = all personals (newest toward the top).
    - A horizontal dotted fork row exists for each personal trained directly on a
      baseline; that baseline and that personal share the same y.
    - Nodes that fall *between* two fork times are spread evenly inside that band
      on their own track (5 baselines vs 2 personals ⇒ different spacing, same band).
    """
    baselines = [item for item in placed if item.kind == "baseline"]
    personals = [item for item in placed if item.kind == "personal"]
    if not baselines and not personals:
        return {}

    baseline_x = float(_MARGIN + _LABEL_W)
    personal_x = baseline_x + _NODE_W + _GAP_X
    baseline_by_version = {item.node.version: item for item in baselines}
    personal_versions = {item.node.version for item in personals}

    forks: list[tuple[_Placed, _Placed]] = []
    fork_baseline_versions: set[str] = set()
    fork_personal_versions: set[str] = set()
    for item in personals:
        if not _is_personal_arm_root(item.node, personal_versions=personal_versions):
            continue
        baseline_key = _baseline_parent_key(item)
        if not baseline_key or baseline_key not in baseline_by_version:
            continue
        forks.append((baseline_by_version[baseline_key], item))
        fork_baseline_versions.add(baseline_key)
        fork_personal_versions.add(item.node.version)
    # Newest fork at the top.
    forks.sort(
        key=lambda pair: (pair[0].node.trained_at, pair[0].node.version),
        reverse=True,
    )

    def _time_key(item: _Placed) -> tuple:
        return (item.node.trained_at, item.node.version)

    non_fork_baselines = sorted(
        [item for item in baselines if item.node.version not in fork_baseline_versions],
        key=_time_key,
        reverse=True,
    )
    non_fork_personals = sorted(
        [item for item in personals if item.node.version not in fork_personal_versions],
        key=_time_key,
        reverse=True,
    )

    # Bands newest→oldest: (y_top_exclusive, y_bottom_exclusive) for interstitials,
    # plus fork rows at the boundaries. Index 0 = above newest fork, then between
    # forks, then below oldest fork.
    band_baselines: list[list[_Placed]] = [[] for _ in range(len(forks) + 1)]
    band_personals: list[list[_Placed]] = [[] for _ in range(len(forks) + 1)]

    def _band_index(trained_at) -> int:
        for index, (baseline_item, _personal_item) in enumerate(forks):
            if trained_at > baseline_item.node.trained_at:
                return index
        return len(forks)

    for item in non_fork_baselines:
        band_baselines[_band_index(item.node.trained_at)].append(item)
    for item in non_fork_personals:
        band_personals[_band_index(item.node.trained_at)].append(item)

    # Vertical budget per band ~ max(track counts) so both tracks span the band.
    def _band_span(n_left: int, n_right: int) -> float:
        count = max(n_left, n_right, 0)
        if count == 0:
            return float(_NODE_H + _GAP_Y)
        return float(count + 1) * (_NODE_H + _GAP_Y)

    baseline_positions: dict[str, float] = {}
    personal_positions: dict[str, float] = {}

    if not forks:
        y_top = float(_MARGIN)
        y_bottom = y_top + _band_span(len(baselines), len(personals))
        baseline_positions = _spread_ys(
            sorted(baselines, key=_time_key, reverse=True), y_top, y_bottom
        )
        personal_positions = _spread_ys(
            sorted(personals, key=_time_key, reverse=True), y_top, y_bottom
        )
    else:
        fork_ys: list[float] = []
        band_bounds: list[tuple[float, float]] = []
        y_cursor = float(_MARGIN)
        # Band above newest fork, then fork row, then band, then fork, ...
        for index, _fork in enumerate(forks):
            y_band_top = y_cursor
            y_cursor += _band_span(len(band_baselines[index]), len(band_personals[index]))
            band_bounds.append((y_band_top, y_cursor))
            fork_ys.append(y_cursor)
            y_cursor += float(_NODE_H + _GAP_Y)
        band_bounds.append((
            y_cursor,
            y_cursor + _band_span(len(band_baselines[len(forks)]), len(band_personals[len(forks)])),
        ))

        for (baseline_item, personal_item), fork_y in zip(forks, fork_ys, strict=True):
            baseline_positions[baseline_item.node.version] = fork_y
            personal_positions[personal_item.node.version] = fork_y
        for index, (y_a, y_b) in enumerate(band_bounds):
            baseline_positions.update(_spread_ys(band_baselines[index], y_a, y_b))
            personal_positions.update(_spread_ys(band_personals[index], y_a, y_b))

    boxes: dict[str, _Box] = {}
    for item in baselines:
        key = _box_key("baseline", item.node.version)
        boxes[key] = _Box(
            key=key,
            kind="baseline",
            node=item.node,
            x=baseline_x,
            y=baseline_positions[item.node.version],
        )
    for item in personals:
        key = _box_key("personal", item.node.version)
        boxes[key] = _Box(
            key=key,
            kind="personal",
            node=item.node,
            x=personal_x,
            y=personal_positions[item.node.version],
            parent_baseline=item.parent_baseline,
        )
    return boxes


def _spread_ys(
    items: list[_Placed],
    y_top: float,
    y_bottom: float,
) -> dict[str, float]:
    """Evenly place nodes inside (y_top, y_bottom), newest first already ordered."""
    if not items:
        return {}
    span = y_bottom - y_top
    if span <= 0:
        return {item.node.version: y_top for item in items}
    n = len(items)
    return {
        item.node.version: y_top + (index + 1) * span / (n + 1) for index, item in enumerate(items)
    }


def _draw_node(
    draw: ImageDraw.ImageDraw,
    box: _Box,
    *,
    node_w: float,
    node_h: float,
    font: ImageFont.ImageFont,
    font_small: ImageFont.ImageFont,
) -> None:
    fill, outline, width = _node_colors(box.node, yours=box.kind == "personal")
    x0, y0 = box.x, box.y
    x1, y1 = box.x + node_w, box.y + node_h
    draw.rounded_rectangle(
        (x0, y0, x1, y1),
        radius=max(6, int(10 * (node_w / _NODE_W))),
        fill=fill,
        outline=outline,
        width=max(1, int(width * (node_w / _NODE_W))),
    )
    title_fill, meta_fill = _text_colors(box.node)
    pad = max(6, int(8 * (node_w / _NODE_W)))
    meta = _short_meta(box.node)
    if meta:
        draw.text((x0 + pad, y0 + pad), _fit_text(box.node.title, 16), fill=title_fill, font=font)
        draw.text(
            (x0 + pad, y0 + pad + 22 * (node_h / _NODE_H)),
            _fit_text(meta, 22),
            fill=meta_fill,
            font=font_small,
        )
    else:
        # Center the version when there is no secondary line.
        draw.text(
            (x0 + pad, y0 + node_h / 2 - 8),
            _fit_text(box.node.title, 16),
            fill=title_fill,
            font=font,
        )


def _draw_edge(
    draw: ImageDraw.ImageDraw,
    src: _Box,
    dst: _Box,
    *,
    style: str,
    node_w: float,
    node_h: float,
) -> None:
    # Parent → child. Newer sits above older on a trunk, so many solid edges go upward.
    if abs(src.x - dst.x) < 1:
        # Same column: connect bottom of upper node to top of lower node (or reverse).
        if src.y < dst.y:
            start = (src.x + node_w / 2, src.y + node_h)
            end = (dst.x + node_w / 2, dst.y)
        else:
            start = (src.x + node_w / 2, src.y)
            end = (dst.x + node_w / 2, dst.y + node_h)
    else:
        # Baseline → personal: right side of baseline to left side of personal.
        start = (src.x + node_w, src.y + node_h / 2)
        end = (dst.x, dst.y + node_h / 2)
    color = (100, 116, 139, 210) if style == "dotted" else (71, 85, 105, 230)
    if style == "dotted":
        _draw_dotted_line(draw, start, end, fill=color, width=2)
    else:
        draw.line([start, end], fill=color, width=2)
    _draw_arrowhead(draw, start, end, fill=color)


def _draw_dotted_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    fill: tuple[int, int, int, int],
    width: int,
) -> None:
    x0, y0 = start
    x1, y1 = end
    segments = 18
    for index in range(segments):
        if index % 2:
            continue
        t0 = index / segments
        t1 = min((index + 1) / segments, 1.0)
        draw.line(
            [
                (x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0),
                (x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1),
            ],
            fill=fill,
            width=width,
        )


def _draw_arrowhead(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    fill: tuple[int, int, int, int],
) -> None:
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 8
    left = (
        end[0] - size * math.cos(angle - 0.4),
        end[1] - size * math.sin(angle - 0.4),
    )
    right = (
        end[0] - size * math.cos(angle + 0.4),
        end[1] - size * math.sin(angle + 0.4),
    )
    draw.polygon([end, left, right], fill=fill)


def _node_colors(
    node: LineageNode, *, yours: bool
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], int]:
    """Fixed status palette (not Settings theme tiles).

    - currently promoted → strong green fill
    - was promoted (archived) → muted sage fill
    - never / not yet promoted (candidate) → neutral slate fill
    - parent baseline (any era) → gold band only
    """
    del yours  # Track is positional; fill is status-only.
    if node.status_key == "promoted":
        fill, outline, width = (22, 163, 74, 255), (21, 128, 61, 255), 2
    elif node.status_key == "candidate":
        fill, outline, width = (248, 250, 252, 255), (148, 163, 184, 255), 2
    else:
        # archived ≈ former production in the normal path
        fill, outline, width = (220, 237, 214, 255), (132, 162, 124, 255), 2
    if node.is_parent:
        return fill, (212, 160, 23, 255), 4
    return fill, outline, width


def _text_colors(
    node: LineageNode,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    if node.status_key == "promoted":
        return (255, 255, 255, 255), (220, 252, 231, 255)
    return (30, 41, 59, 255), (71, 85, 105, 255)


def _short_meta(node: LineageNode) -> str:
    # Status/parent encoded by fill + gold band; only keep weight-origin hint.
    if node.parent_version and node.parent_kind == _BASELINE_PARENT_KIND:
        return f"from {node.parent_version}"
    if node.parent_version:
        return f"from {node.parent_version}"
    return ""


def _fit_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _box_key(kind: str, version: str) -> str:
    return f"{kind}:{version}"


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


def _is_personal_arm_root(node: LineageNode, *, personal_versions: set[str]) -> bool:
    if node.parent_kind == _PERSONAL_PARENT_KIND and node.parent_version in personal_versions:
        return False
    return True


def _baseline_parent_key(item: _Placed) -> str | None:
    node = item.node
    if node.parent_kind == _BASELINE_PARENT_KIND and node.parent_version:
        return node.parent_version
    return item.parent_baseline
