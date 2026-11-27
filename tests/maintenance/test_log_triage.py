"""Unit tests for maintenance.log_triage helpers."""

from __future__ import annotations

import json

import pytest

from chess_teacher.maintenance.log_triage.decide import DecisionInput, decide
from chess_teacher.maintenance.log_triage.eval_ids import make_eval_id
from chess_teacher.maintenance.log_triage.grace import is_grace_blocked, make_grace_entry
from chess_teacher.maintenance.log_triage.redact import redact_snippet
from chess_teacher.maintenance.log_triage.schemas import (
    StateValidationError,
    empty_global_known,
    empty_source_state,
    validate_source_state,
)
from chess_teacher.maintenance.log_triage.signature import (
    coarse_signature_from_gha,
    coarse_signature_from_log,
)
from chess_teacher.maintenance.log_triage.state_store import (
    find_global_by_signature,
    upsert_global_issue,
)


def test_coarse_signature_stable_across_uuids() -> None:
    base = {
        "logger": "chess_teacher.foo",
        "exc_type": "ValueError",
        "traceback": 'File "/app/src/chess_teacher/foo/bar.py", line 10, in do_x\n  raise ValueError',
        "msg": "failed id=aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    }
    a = coarse_signature_from_log(base, source="s3")
    b = coarse_signature_from_log({**base, "log_id": "other"}, source="s3")
    assert a == b
    assert a.startswith("v1|s3|")
    assert "chess_teacher.foo.bar" in a or "foo.bar" in a


def test_gha_signature() -> None:
    sig = coarse_signature_from_gha(workflow="ci.yml", job="test", step="Run pytest")
    assert sig == "v1|gha|ci.yml|test|Run pytest"


def test_redact_email_and_token() -> None:
    text = "user=a@b.com token=supersecretvalue password=hunter2"
    out = redact_snippet(text)
    assert "a@b.com" not in out
    assert "<email>" in out
    assert "supersecretvalue" not in out


def test_decide_act_and_hold() -> None:
    act = decide(DecisionInput(action="open", confidence=0.8, open_threshold=0.7, acts_so_far=0))
    assert act.decision == "act"
    hold = decide(DecisionInput(action="open", confidence=0.5, open_threshold=0.7))
    assert hold.decision == "hold"
    grace = decide(
        DecisionInput(
            action="reopen",
            confidence=0.95,
            open_threshold=0.7,
            grace_blocked=True,
        )
    )
    assert grace.decision == "hold"
    caps = decide(
        DecisionInput(
            action="open",
            confidence=0.99,
            open_threshold=0.7,
            acts_so_far=5,
            max_acts=5,
        )
    )
    assert caps.decision == "hold"


def test_eval_id_stable() -> None:
    a = make_eval_id(
        run_id="r1",
        signature="v1|s3|a|b|c",
        action="open",
        scan_from="t0",
        scan_to="t1",
    )
    b = make_eval_id(
        run_id="r1",
        signature="v1|s3|a|b|c",
        action="open",
        scan_from="t0",
        scan_to="t1",
    )
    assert a == b
    assert len(a) == 16


def test_validate_empty_state() -> None:
    doc = empty_source_state(source="s3")
    validate_source_state(doc)
    bad = dict(doc)
    bad["source"] = "nope"
    with pytest.raises(StateValidationError):
        validate_source_state(bad)


def test_grace_blocked() -> None:
    entry = make_grace_entry(
        issue=1,
        signature="v1|s3|a|b|c",
        grace_hours=6,
        reason="test",
    )
    blocked = is_grace_blocked([entry], signature="v1|s3|a|b|c")
    assert blocked["blocked"] is True
    clear = is_grace_blocked([entry], signature="other")
    assert clear["blocked"] is False


def test_global_upsert() -> None:
    doc = empty_global_known()
    upsert_global_issue(
        doc,
        issue=7,
        signature="sig",
        status="open",
        source="s3",
        updated_at="2026-01-01T00:00:00Z",
    )
    upsert_global_issue(
        doc,
        issue=7,
        signature="sig",
        status="pr_open",
        source="gha",
        updated_at="2026-01-02T00:00:00Z",
    )
    hit = find_global_by_signature(doc, "sig")
    assert hit is not None
    assert hit["status"] == "pr_open"
    assert set(hit["sources"]) == {"s3", "gha"}


def test_state_roundtrip_json() -> None:
    doc = empty_source_state(source="gha")
    raw = json.dumps(doc)
    loaded = json.loads(raw)
    validate_source_state(loaded)
