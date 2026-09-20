"""Candidate MultiPV JSON shape (same contract as ``candidate_eval``)."""

from __future__ import annotations

from typing import Any

PAYLOAD_KEY_DEPTH = "depth"
PAYLOAD_KEY_NODES = "num_nodes"
PAYLOAD_KEY_METHOD = "method"
PAYLOAD_KEY_EVALS = "evals_white_pov"
CANDIDATE_SEARCH_METHOD = "multipv_nodes"


def build_candidate_payload(
    evals_white_pov: dict[str, float],
    *,
    depth: int,
    num_nodes: int,
    method: str = CANDIDATE_SEARCH_METHOD,
) -> dict[str, Any]:
    return {
        PAYLOAD_KEY_DEPTH: int(depth),
        PAYLOAD_KEY_METHOD: method,
        PAYLOAD_KEY_EVALS: {str(key): float(value) for key, value in evals_white_pov.items()},
        PAYLOAD_KEY_NODES: int(num_nodes),
    }


def payload_evals(payload: dict[str, Any] | None) -> dict[str, float]:
    if not payload:
        return {}
    raw = payload.get(PAYLOAD_KEY_EVALS)
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for uci, value in raw.items():
        try:
            out[str(uci)] = float(value)
        except (TypeError, ValueError):
            continue
    return out
