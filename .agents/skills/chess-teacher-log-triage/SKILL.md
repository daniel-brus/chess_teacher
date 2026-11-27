---
name: chess-teacher-log-triage
description: >-
  Deterministic helpers for the agent log→issue→PR pipeline: triage state
  (watermark/handoff/known_issues/grace), coarse signatures, confidence gate,
  audit writes, locks, GitHub ensure-issue, Slack message formatting. Use when
  running or debugging Cursor Triage/Implement Automations, or local dry-runs.
---

# Chess Teacher Log Triage

Durable control plane for triage Automations. **Helpers are deterministic — no LLM inside CLIs.**

## Doppler

Use config **`prod_agent_triage`** (Doppler requires the `prod_` prefix under the Production environment; this is the plan’s `agent_triage` config).

See [DOPPLER.md](DOPPLER.md). **Cursor Automations:** set `DOPPLER_TOKEN_TRIAGE` (service token → `prod_agent_triage`); prompts map it to `DOPPLER_TOKEN=…` for the CLI. Leave general `DOPPLER_TOKEN` unchanged.

```bash
doppler run --project chess-teacher --config prod_agent_triage -- .venv/Scripts/python.exe .agents/skills/chess-teacher-log-triage/scripts/triage_query.py --json info
```

Local MinIO smoke can use `dev_local` for storage I/O tests (knobs fall back to defaults if unset).

## Storage layout

```text
agent/triage/state/{s3,vps,gha}.json
agent/triage/state/known_issues_global.json
agent/triage/locks/{s3,vps,gha}.lock.json
agent/triage/audit/YYYY/MM/DD/{eval_id}.json
```

## CLI (`triage_query.py`)

Always `--json`. From repo root with venv. Important subcommands:

| Command | Purpose |
|---------|---------|
| `info` / `settings` | Prefix + TRIAGE_* knobs |
| `provenance` | Git branch/SHA (+ optional observed deploy) |
| `signature --record '{...}'` | Coarse log signature (`--gha` for Actions) |
| `decide --action open --confidence 0.8` | act/hold gate |
| `state-init` / `state-load` / `state-save` | Per-source state |
| `global-load` | Cross-source known issues |
| `lock --source s3` / `--release` | Single-flight lock |
| `eval-write --evaluation '{...}'` | Audit + Slack text |
| `grace-check` / `grace-add` | Post-merge suppress window |
| `find-issue` / `ensure-issue` | GitHub create/comment/reopen |

Library code: `chess_teacher.maintenance.log_triage`.

## Slack

Channel `#chess-teacher-triage` (`C0CMEAGLX61`) — see [SLACK.md](SLACK.md).

## Automation contract (summary)

1. Load state + locks; scan logs since `watermark_ts` (via **chess-teacher-logs** / GHA `gh`).
2. For each open/reopen candidate: scout → High/Low/Judge debate → `eval-write` → Slack → `decide`.
3. On `act`: `ensure-issue` with labels `from-logs|from-ci|from-vps` + `agent-fix`.
4. Rewrite verbose handoff; advance watermark only if `source_ok`.

Implement Automation (separate): on `agent-fix` → claim `in-progress` → PR to **`develop`** with `Fixes #N` → never merge.

## VPS source

Enabled. `prod_agent_triage` includes `DEPLOY_*`; Triage uses chess-teacher-vps (`ping` / `image-tag` / `pods` / `logs`). See [VPS.md](VPS.md).

## Implement trigger

Issue label `agent-fix` → GitHub Action posts to Cursor Implement webhook. See [IMPLEMENT_WEBHOOK.md](IMPLEMENT_WEBHOOK.md).

## Related skills

- **chess-teacher-logs** — ERROR search S3/local
- **chess-teacher-storage** — inspect `agent/triage/`
- **chess-teacher-vps** — kubectl logs (v2)
- **chess-teacher-merge-develop** — human merge only
