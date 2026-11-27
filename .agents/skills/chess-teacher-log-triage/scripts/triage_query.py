#!/usr/bin/env python3
"""CLI for agent log triage helpers (state, locks, decide, audit, issues)."""

from __future__ import annotations

import os

os.environ.setdefault("ENVIRONMENT", "AGENT")

import argparse
import json
import uuid
from datetime import UTC, datetime
from typing import Any

from chess_teacher.maintenance.log_triage.audit import write_eval_audit
from chess_teacher.maintenance.log_triage.constants import SOURCES, TRIAGE_PREFIX
from chess_teacher.maintenance.log_triage.decide import DecisionInput, decide
from chess_teacher.maintenance.log_triage.ensure_issue import (
    build_issue_body,
    ensure_issue,
    find_issues_by_signature,
)
from chess_teacher.maintenance.log_triage.eval_ids import make_eval_id
from chess_teacher.maintenance.log_triage.grace import is_grace_blocked, make_grace_entry
from chess_teacher.maintenance.log_triage.locks import acquire_lock, release_lock
from chess_teacher.maintenance.log_triage.provenance import collect_provenance
from chess_teacher.maintenance.log_triage.redact import redact_snippet
from chess_teacher.maintenance.log_triage.settings import load_settings
from chess_teacher.maintenance.log_triage.signature import (
    coarse_signature_from_gha,
    coarse_signature_from_log,
)
from chess_teacher.maintenance.log_triage.slack_format import (
    format_slack_eval,
    format_slack_run_summary,
)
from chess_teacher.maintenance.log_triage.state_store import (
    find_global_by_signature,
    load_global_known,
    load_source_state,
    save_global_known,
    save_source_state,
    state_key,
    upsert_global_issue,
)
from chess_teacher.utils.object_storage.factory import get_raw_storage


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _parse_json_arg(raw: str | None) -> Any:
    if raw is None:
        return None
    if raw == "-":
        import sys

        return json.load(sys.stdin)
    return json.loads(raw)


def cmd_info(_: argparse.Namespace) -> dict[str, Any]:
    settings = load_settings()
    return {
        "triage_prefix": TRIAGE_PREFIX,
        "doppler_config_hint": "prod_agent_triage",
        "sources": sorted(SOURCES),
        "settings": settings.__dict__,
    }


def cmd_provenance(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "ok": True,
        "provenance": collect_provenance(
            observed_ref=args.observed_ref,
            observed_image_tag=args.observed_image_tag,
            observed_deploy_at=args.observed_deploy_at,
        ),
    }


def cmd_signature(args: argparse.Namespace) -> dict[str, Any]:
    if args.gha:
        payload = _parse_json_arg(args.record) or {}
        sig = coarse_signature_from_gha(
            workflow=str(payload.get("workflow") or args.workflow or ""),
            job=str(payload.get("job") or args.job or ""),
            step=payload.get("step") or args.step,
        )
        return {"ok": True, "signature": sig, "kind": "gha"}
    record = _parse_json_arg(args.record)
    if not isinstance(record, dict):
        return {"ok": False, "error": "provide --record JSON object (or - for stdin)"}
    sig = coarse_signature_from_log(record, source=args.source)
    return {"ok": True, "signature": sig, "kind": "log"}


def cmd_redact(args: argparse.Namespace) -> dict[str, Any]:
    return {"ok": True, "redacted": redact_snippet(args.text, max_len=args.max_len)}


def cmd_decide(args: argparse.Namespace) -> dict[str, Any]:
    settings = load_settings()
    result = decide(
        DecisionInput(
            action=args.action,
            confidence=args.confidence,
            open_threshold=args.threshold
            if args.threshold is not None
            else settings.open_threshold,
            grace_blocked=args.grace_blocked,
            flap_locked=args.flap_locked,
            acts_so_far=args.acts_so_far,
            max_acts=args.max_acts if args.max_acts is not None else settings.max_acts,
            evals_so_far=args.evals_so_far,
            max_evals=args.max_evals if args.max_evals is not None else settings.max_evals,
        )
    )
    return {
        "ok": True,
        "decision": result.decision,
        "reasons": result.reasons,
        "open_threshold": args.threshold if args.threshold is not None else settings.open_threshold,
    }


def cmd_state_load(args: argparse.Namespace) -> dict[str, Any]:
    storage = get_raw_storage()
    doc = load_source_state(storage, args.source, create_if_missing=not args.require_exists)
    return {"ok": True, "key": state_key(args.source), "state": doc}


def cmd_state_save(args: argparse.Namespace) -> dict[str, Any]:
    storage = get_raw_storage()
    doc = _parse_json_arg(args.document)
    if not isinstance(doc, dict):
        return {"ok": False, "error": "--document must be JSON object"}
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    expected = args.expected_updated_at or doc.get("updated_at")
    doc["updated_at"] = now
    saved = save_source_state(storage, doc, expected_updated_at=expected)
    return {"ok": True, "key": state_key(saved["source"]), "state": saved}


def cmd_state_init(args: argparse.Namespace) -> dict[str, Any]:
    storage = get_raw_storage()
    from chess_teacher.maintenance.log_triage.schemas import empty_source_state

    results = []
    sources = [args.source] if args.source else sorted(SOURCES)
    for source in sources:
        doc = empty_source_state(source=source)
        if args.force:
            storage.write_text_atomic(
                state_key(source),
                json.dumps(doc, indent=2) + "\n",
                overwrite=True,
            )
            results.append({"source": source, "created": True, "forced": True})
            continue
        existing = storage.read_bytes(state_key(source))
        if existing is not None:
            results.append({"source": source, "created": False, "exists": True})
            continue
        save_source_state(storage, doc, expected_updated_at=None)
        results.append({"source": source, "created": True})
    global_doc = load_global_known(storage, create_if_missing=True)
    save_global_known(storage, global_doc)
    return {
        "ok": True,
        "results": results,
        "global_known_key": "agent/triage/state/known_issues_global.json",
    }


def cmd_global_load(_: argparse.Namespace) -> dict[str, Any]:
    storage = get_raw_storage()
    return {"ok": True, "global": load_global_known(storage)}


def cmd_lock(args: argparse.Namespace) -> dict[str, Any]:
    storage = get_raw_storage()
    settings = load_settings()
    holder = args.holder or f"triage-{uuid.uuid4().hex[:8]}"
    if args.release:
        return {"ok": True, **release_lock(storage, args.source, holder=holder)}
    result = acquire_lock(
        storage,
        args.source,
        holder=holder,
        ttl_minutes=args.ttl_minutes or settings.lock_ttl_minutes,
    )
    return {"ok": True, **result, "holder": holder}


def cmd_eval_write(args: argparse.Namespace) -> dict[str, Any]:
    storage = get_raw_storage()
    evaluation = _parse_json_arg(args.evaluation)
    if not isinstance(evaluation, dict):
        return {"ok": False, "error": "--evaluation must be JSON object"}
    if "eval_id" not in evaluation:
        evaluation["eval_id"] = make_eval_id(
            run_id=str(evaluation.get("run_id") or "run"),
            signature=str(evaluation.get("signature") or ""),
            action=str(evaluation.get("action") or ""),
            scan_from=str(evaluation.get("scan_from") or ""),
            scan_to=str(evaluation.get("scan_to") or ""),
        )
    written = write_eval_audit(storage, evaluation)
    return {"ok": True, **written, "slack_text": format_slack_eval(written["evaluation"])}


def cmd_eval_id(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "ok": True,
        "eval_id": make_eval_id(
            run_id=args.run_id,
            signature=args.signature,
            action=args.action,
            scan_from=args.scan_from,
            scan_to=args.scan_to,
        ),
    }


def cmd_format_slack(args: argparse.Namespace) -> dict[str, Any]:
    payload = _parse_json_arg(args.payload)
    if not isinstance(payload, dict):
        return {"ok": False, "error": "--payload JSON required"}
    if args.kind == "summary":
        return {"ok": True, "text": format_slack_run_summary(payload)}
    return {"ok": True, "text": format_slack_eval(payload)}


def cmd_grace_check(args: argparse.Namespace) -> dict[str, Any]:
    storage = get_raw_storage()
    state = load_source_state(storage, args.source)
    return {
        "ok": True,
        **is_grace_blocked(state.get("grace") or [], signature=args.signature),
    }


def cmd_grace_add(args: argparse.Namespace) -> dict[str, Any]:
    storage = get_raw_storage()
    settings = load_settings()
    state = load_source_state(storage, args.source)
    expected = state["updated_at"]
    entry = make_grace_entry(
        issue=args.issue,
        signature=args.signature,
        grace_hours=args.hours or settings.grace_hours,
        reason=args.reason or "post-merge grace",
    )
    grace = [g for g in (state.get("grace") or []) if g.get("signature") != args.signature]
    grace.append(entry)
    state["grace"] = grace
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    state["updated_at"] = now
    saved = save_source_state(storage, state, expected_updated_at=expected)
    return {"ok": True, "entry": entry, "state": saved}


def cmd_find_issue(args: argparse.Namespace) -> dict[str, Any]:
    settings = load_settings()
    storage = get_raw_storage()
    global_doc = load_global_known(storage)
    global_hit = find_global_by_signature(global_doc, args.signature)
    gh_result = find_issues_by_signature(
        args.signature,
        lookback_days=args.lookback_days or settings.closed_lookback_days,
        include_closed=not args.open_only,
    )
    return {"ok": True, "global_hit": global_hit, "github": gh_result}


def cmd_ensure_issue(args: argparse.Namespace) -> dict[str, Any]:
    body = args.body
    if not body:
        body = build_issue_body(
            signature=args.signature,
            title_hint=args.title,
            rationale=args.rationale or "",
            eval_id=args.eval_id or "manual",
            provenance=collect_provenance(),
            evidence=_parse_json_arg(args.evidence) or [],
        )
    labels = [x for x in (args.labels or "").split(",") if x]
    result = ensure_issue(
        action=args.action,
        signature=args.signature,
        title=args.title,
        body=body,
        labels=labels,
        existing_issue=args.issue,
        comment=args.comment,
        dry_run=args.dry_run,
    )
    if result.get("ok") and not args.dry_run and args.action == "create" and args.upsert_global:
        # Best-effort: parse issue number from URL if present
        storage = get_raw_storage()
        global_doc = load_global_known(storage)
        url = str(result.get("url") or "")
        issue_no = args.issue
        if issue_no is None and "/issues/" in url:
            try:
                issue_no = int(url.rstrip("/").split("/")[-1])
            except ValueError:
                issue_no = None
        if issue_no is not None:
            now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            upsert_global_issue(
                global_doc,
                issue=issue_no,
                signature=args.signature,
                status="open",
                source=args.source or "s3",
                updated_at=now,
            )
            save_global_known(storage, global_doc)
            result["issue"] = issue_no
    return result


def cmd_settings(_: argparse.Namespace) -> dict[str, Any]:
    return {"ok": True, "settings": load_settings().__dict__}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="JSON output (always on)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="Triage prefix + settings")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("settings", help="Show TRIAGE_* settings")
    p.set_defaults(func=cmd_settings)

    p = sub.add_parser("provenance", help="Collect git/deploy provenance")
    p.add_argument("--observed-ref")
    p.add_argument("--observed-image-tag")
    p.add_argument("--observed-deploy-at")
    p.set_defaults(func=cmd_provenance)

    p = sub.add_parser("signature", help="Compute coarse signature")
    p.add_argument("--record", help="JSON log record or GHA fields")
    p.add_argument("--source", default="s3", choices=sorted(SOURCES))
    p.add_argument("--gha", action="store_true")
    p.add_argument("--workflow")
    p.add_argument("--job")
    p.add_argument("--step")
    p.set_defaults(func=cmd_signature)

    p = sub.add_parser("redact", help="Redact a snippet")
    p.add_argument("--text", required=True)
    p.add_argument("--max-len", type=int, default=400)
    p.set_defaults(func=cmd_redact)

    p = sub.add_parser("decide", help="Run act/hold gate")
    p.add_argument("--action", required=True)
    p.add_argument("--confidence", type=float, required=True)
    p.add_argument("--threshold", type=float)
    p.add_argument("--grace-blocked", action="store_true")
    p.add_argument("--flap-locked", action="store_true")
    p.add_argument("--acts-so-far", type=int, default=0)
    p.add_argument("--max-acts", type=int)
    p.add_argument("--evals-so-far", type=int, default=0)
    p.add_argument("--max-evals", type=int)
    p.set_defaults(func=cmd_decide)

    p = sub.add_parser("state-load", help="Load per-source state")
    p.add_argument("--source", required=True, choices=sorted(SOURCES))
    p.add_argument("--require-exists", action="store_true")
    p.set_defaults(func=cmd_state_load)

    p = sub.add_parser("state-save", help="Save per-source state (RMW)")
    p.add_argument("--document", required=True, help="JSON state or - for stdin")
    p.add_argument("--expected-updated-at")
    p.set_defaults(func=cmd_state_save)

    p = sub.add_parser("state-init", help="Create empty state objects if missing")
    p.add_argument("--source", choices=sorted(SOURCES))
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_state_init)

    p = sub.add_parser("global-load", help="Load known_issues_global")
    p.set_defaults(func=cmd_global_load)

    p = sub.add_parser("lock", help="Acquire or release source lock")
    p.add_argument("--source", required=True, choices=sorted(SOURCES))
    p.add_argument("--holder")
    p.add_argument("--ttl-minutes", type=int)
    p.add_argument("--release", action="store_true")
    p.set_defaults(func=cmd_lock)

    p = sub.add_parser("eval-id", help="Compute stable eval id")
    p.add_argument("--run-id", required=True)
    p.add_argument("--signature", required=True)
    p.add_argument("--action", required=True)
    p.add_argument("--scan-from", required=True)
    p.add_argument("--scan-to", required=True)
    p.set_defaults(func=cmd_eval_id)

    p = sub.add_parser("eval-write", help="Write audit eval + Slack text")
    p.add_argument("--evaluation", required=True, help="JSON eval or -")
    p.set_defaults(func=cmd_eval_write)

    p = sub.add_parser("format-slack", help="Format Slack text from JSON")
    p.add_argument("--payload", required=True)
    p.add_argument("--kind", choices=["eval", "summary"], default="eval")
    p.set_defaults(func=cmd_format_slack)

    p = sub.add_parser("grace-check", help="Check grace for signature")
    p.add_argument("--source", required=True, choices=sorted(SOURCES))
    p.add_argument("--signature", required=True)
    p.set_defaults(func=cmd_grace_check)

    p = sub.add_parser("grace-add", help="Add/replace grace entry in state")
    p.add_argument("--source", required=True, choices=sorted(SOURCES))
    p.add_argument("--issue", type=int, required=True)
    p.add_argument("--signature", required=True)
    p.add_argument("--hours", type=int)
    p.add_argument("--reason")
    p.set_defaults(func=cmd_grace_add)

    p = sub.add_parser("find-issue", help="Find issue by signature (global + gh)")
    p.add_argument("--signature", required=True)
    p.add_argument("--lookback-days", type=int)
    p.add_argument("--open-only", action="store_true")
    p.set_defaults(func=cmd_find_issue)

    p = sub.add_parser("ensure-issue", help="create|comment|reopen via gh")
    p.add_argument("--action", required=True, choices=["create", "comment", "reopen"])
    p.add_argument("--signature", required=True)
    p.add_argument("--title", default="Agent triage finding")
    p.add_argument("--body")
    p.add_argument("--rationale")
    p.add_argument("--eval-id")
    p.add_argument("--evidence", help="JSON list")
    p.add_argument("--labels", default="from-logs,agent-fix")
    p.add_argument("--issue", type=int)
    p.add_argument("--comment")
    p.add_argument("--source", choices=sorted(SOURCES))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--upsert-global", action="store_true")
    p.set_defaults(func=cmd_ensure_issue)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        payload = args.func(args)
    except Exception as exc:
        _emit({"ok": False, "error": str(exc), "error_type": type(exc).__name__})
        raise SystemExit(1) from exc
    _emit(payload)
    if isinstance(payload, dict) and payload.get("ok") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
