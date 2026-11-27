"""Format Slack message bodies for triage evals and run summaries."""

from __future__ import annotations

from typing import Any


def format_slack_eval(evaluation: dict[str, Any]) -> str:
    conf = evaluation.get("confidence")
    threshold = evaluation.get("open_threshold")
    decision = evaluation.get("decision")
    action = evaluation.get("action")
    source = evaluation.get("source")
    eval_id = evaluation.get("eval_id")
    issue = evaluation.get("issue")
    signature = evaluation.get("signature")
    rationale = evaluation.get("rationale") or ""
    prov = evaluation.get("provenance") or {}
    triage_sha = (prov.get("triage_git_sha") or "?")[:7]
    triage_branch = prov.get("triage_git_branch") or "?"
    observed = prov.get("observed_ref") or "main@unknown"

    lines = [
        f"*Triage eval* `{eval_id}` · source=`{source}`",
        f"action=`{action}` decision=`*{decision}*` confidence=`{conf}` threshold=`{threshold}`",
        f"signature=`{signature}`",
        f"issue={issue if issue is not None else 'new'}",
        f"triage@{triage_sha} ({triage_branch}) observing `{observed}`",
        "",
        rationale[:500],
    ]
    debate = evaluation.get("debate")
    if isinstance(debate, dict):
        lines.extend([
            "",
            f"_debate_ prelim={debate.get('preliminary_confidence')} → "
            f"final={debate.get('final_confidence')}",
            f"High≈{((debate.get('advocate_high') or {}).get('confidence_claim'))} "
            f"Low≈{((debate.get('advocate_low') or {}).get('confidence_claim'))}",
        ])
    return "\n".join(lines).strip()


def format_slack_run_summary(summary: dict[str, Any]) -> str:
    return (
        f"*Triage run* `{summary.get('run_id')}`\n"
        f"sources={summary.get('sources')} "
        f"evals={summary.get('evals')} acts={summary.get('acts')} "
        f"holds={summary.get('holds')} caps_hit={summary.get('caps_hit')}\n"
        f"triage@{str(summary.get('triage_git_sha') or '?')[:7]} "
        f"({summary.get('triage_git_branch')})"
    )
