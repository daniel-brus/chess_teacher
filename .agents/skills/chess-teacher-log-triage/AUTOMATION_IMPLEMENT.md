# Cursor Automation: Implement (prompt + wiring)

## Trigger

GitHub issue **label added:** `agent-fix`
Repo: `chess_teacher`

## Tools

- Slack post → `#chess-teacher-triage`
- GitHub / `gh` / repo checkout

## Instructions

```text
You implement fixes for chess_teacher issues labeled agent-fix.

1. If the issue lacks agent-fix, exit.
2. Remove agent-fix; add in-progress. If that fails because another run claimed it, exit.
3. Search open PRs referencing this issue; if one exists, comment and exit.
4. Create branch agent/issue-<N>-short-slug from develop (or main's merge-base — prefer develop as base for PR).
5. Implement the minimal fix described in the issue. Deny-list without human escalation:
   - DB migrations / schema drops
   - secrets / Doppler / credential files
   - CD manifests targeting production alone
   - force-push / merge
6. Open a PR to base develop with body containing:
   Fixes #<N>
   Fingerprint / eval_id from the issue
   triage provenance if present
7. Add label pr-open on the issue; comment with PR URL.
8. Slack #chess-teacher-triage that the PR is ready.
9. NEVER merge. NEVER push to main.
```

## After merge (human)

Human merges via chess-teacher-merge-develop. Next Triage run should `grace-add` for the signature (`TRIAGE_GRACE_HOURS`).
