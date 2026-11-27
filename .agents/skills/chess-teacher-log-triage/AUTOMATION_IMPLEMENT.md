# Cursor Automation: Implement (prompt + wiring)

## Trigger

GitHub issue label **`agent-fix`** → GHA workflow posts Cursor **webhook** (see [IMPLEMENT_WEBHOOK.md](IMPLEMENT_WEBHOOK.md)). Manual Test/curl also OK.

Repo: `chess_teacher`
Checkout branch: **`develop`**

## Tools

- Slack post → `#chess-teacher-triage` (`C0CMEAGLX61`)
- GitHub / `gh` / repo checkout
- Cursor secret: **`DOPPLER_TOKEN_TRIAGE`** only if you need triage helpers mid-run (usually not required for Implement)

## Instructions (paste into Automation prompt)

```text
You implement fixes for chess_teacher issues labeled agent-fix. Follow .agents/skills/chess-teacher-log-triage/AUTOMATION_IMPLEMENT.md.

1. If the issue lacks agent-fix, exit.
2. Remove agent-fix; add in-progress. If another run claimed it, exit.
3. Search open PRs referencing this issue; if one exists, comment and exit.
4. Create branch agent/issue-<N>-short-slug from latest develop; open PR with base develop.
5. Implement the minimal fix. Deny-list without human escalation: DB migrations/schema drops, secrets/Doppler/credentials, CD-only prod manifests, force-push, merge.
6. PR body must include Fixes #<N>, fingerprint/eval_id from the issue, triage provenance if present.
7. Add label pr-open on the issue; comment with the PR URL.
8. Post to Slack channel C0CMEAGLX61 (#chess-teacher-triage) that the PR is ready.
9. NEVER merge. NEVER push to main.

If you must call triage_query.py, use:
  DOPPLER_TOKEN="$DOPPLER_TOKEN_TRIAGE" doppler run --project chess-teacher --config prod_agent_triage -- \
    .venv/bin/python .agents/skills/chess-teacher-log-triage/scripts/triage_query.py --json <cmd>
```

## After merge (human)

Human merges via chess-teacher-merge-develop. Next Triage run should grace-add for the signature (TRIAGE_GRACE_HOURS).
