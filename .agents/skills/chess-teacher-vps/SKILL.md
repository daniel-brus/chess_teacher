---
name: chess-teacher-vps
description: >-
  Read-only inspection of the production Hetzner VPS (k3s/k8s) via SSH.
  Whitelisted kubectl get/describe/logs, cluster-info, host df/free/uptime,
  and prod Postgres reads via kubectl exec into streamlit
  (scripts/tools/agent_db_query.py). Never apply, delete, restart, or
  ad-hoc shell. SSH credentials from Doppler ci config. Use when the user
  asks about production pods, deploy status, ingress, logs, VPS health, or
  production database contents behind the firewall.
---

# Chess Teacher VPS / Hetzner (read-only)

## ⚠️ Production safety

This is the **live production Hetzner VPS** running k8s (`chess-teacher` namespace). Treat every action as high-risk.

**The agent MUST:**

- Use **only** `.agents/skills/chess-teacher-vps/scripts/vps_query.py` subcommands — fixed, whitelisted remote commands.
- Run **read-only** operations: `kubectl get`, `kubectl describe`, `kubectl logs`, `kubectl cluster-info`, `kubectl rollout status` (status only, no restart), and prod DB reads via `kubectl exec` into `deploy/streamlit` running `python scripts/tools/agent_db_query.py`.
- Prefer **`info`** and **`ping`** before deeper inspection.

**The agent MUST NOT:**

- Run arbitrary SSH commands, interactive shells, or pass user text as remote shell.
- Run `kubectl apply`, `delete`, `patch`, `scale`, `rollout restart`, `port-forward`, or edit manifests on the VPS.
- Run mutating DB ops, backfill, baseline reset/train, or anything that writes to Postgres.
- Run `systemctl stop/restart`, `docker rm/stop`, `rm`, `reboot`, `doppler secrets set`, or anything that changes state.
- Fetch or print Doppler **secret values** from the VPS (only SSH connection metadata via `info`).
- Deploy, sync, or run `apply.sh` unless the user **explicitly** requests a deploy in a separate, deliberate step.

If the user needs a mutating change (restart pod, deploy, config edit), **stop and ask for explicit confirmation** — do not use this skill.

**Long prod ops scripts** (backfill, baseline reset/train) are **not** run via this skill. The user runs them manually on the VPS with `scripts/utils/run_script_job.py`, which renders `orchestration/k8s/job/script.yaml` and `kubectl apply`s a one-off Job (same image + `chess-teacher-env` as streamlit). The agent may document that flow or read Job logs if the user pastes output, but must not apply Jobs through `vps_query.py`.

## Doppler environments

| Config | Role | Used by this skill? |
|--------|------|---------------------|
| `dev_local` | Local Compose + optional k3d staging | No — use kubectl locally for k3d |
| `ci` | GitHub Actions + **SSH to VPS** (`DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_SSH_KEY`) | **Yes** — script reads SSH from here |
| `prod` | Runtime secrets on VPS (Postgres, S3, Redis URLs for the app) | No for SSH; DB queries use env already injected into the streamlit pod |

The VPS script loads SSH credentials from Doppler **`ci`** automatically. Requires `doppler login` with access to project `chess-teacher`.

Runtime app config on the cluster comes from **`prod`** via `doppler run` during CD (`orchestration/k8s/apply.sh`). Do not dump prod secrets through this skill.

## Who runs what

- **The user asks questions in chat** (e.g. “are streamlit pods healthy?” or “how many prod move_characteristics have candidate_evaluations?”).
- **The agent runs** `vps_query.py` subcommands, interprets JSON (`stdout` / `stderr` / `result`), and answers in plain language.
- The script sets **`ENVIRONMENT=AGENT`**. SSH key is written to a temp file with `0600` and deleted immediately after use.

## Rules

- **Read-only only.** No exceptions without explicit user approval outside this skill.
- Always pass **`--json`**. Run from **repository root**. Do not use `uv`.
- Do not run `pytest` / `mypy`; ask the user to run those manually.
- Log / DB output may contain user data — summarize; do not paste large dumps unless asked.
- For **local/dev** Postgres use **chess-teacher-db**. For **prod** Postgres (firewalled) use this skill’s `db-*` commands.

## Agent workflow

1. `info` → confirm target host (no secrets).
2. `ping` → SSH works.
3. Pick command (pods, logs, `db-count`, …).

```bash
python .agents/skills/chess-teacher-vps/scripts/vps_query.py --json info
python .agents/skills/chess-teacher-vps/scripts/vps_query.py --json ping
python .agents/skills/chess-teacher-vps/scripts/vps_query.py --json pods
python .agents/skills/chess-teacher-vps/scripts/vps_query.py --json logs streamlit-xxxxx-yyyyy --tail 50
python .agents/skills/chess-teacher-vps/scripts/vps_query.py --json db-count pipelines/preprocessing move_characteristics --where "candidate_evaluations IS NOT NULL"
```

On Windows (PowerShell): `.venv\Scripts\python.exe` optional for host-only commands; needs `ssh` and `doppler` on PATH. The script writes the deploy key under `%LOCALAPPDATA%\chess-teacher-agent\` with LF newlines and locked-down ACLs — required because OpenSSH rejects keys in `%TEMP%` when Cursor/sandbox adds extra principals.

## Script commands

`python .agents/skills/chess-teacher-vps/scripts/vps_query.py --json <command> ...`

| Command | Remote action (fixed) |
|---------|------------------------|
| `info` | Local: host, user, namespace, safety notice |
| `ping` | `echo ok` |
| `cluster-info` | `kubectl cluster-info` |
| `nodes` | `kubectl get nodes -o wide` |
| `pods` | `kubectl get pods -n chess-teacher -o wide` |
| `deployments` | `kubectl get deployments -n chess-teacher` |
| `services` | `kubectl get svc -n chess-teacher` |
| `ingress` | `kubectl get ingress -n chess-teacher` |
| `events [--tail N]` | Recent namespace events |
| `describe-pod <pod>` | `kubectl describe pod` |
| `logs <pod> [--container C] [--tail N]` | `kubectl logs --tail` |
| `rollout-status` | `kubectl rollout status` (read-only, short timeout) |
| `disk` | `df -h` |
| `memory` | `free -m` |
| `uptime` | `uptime` |
| `db-list-domains` | `kubectl exec deploy/streamlit -- python scripts/tools/agent_db_query.py --json list-domains` |
| `db-list-tables <domain>` | Same via `list-tables` |
| `db-count <domain> <table> [--where EXPR]` | Same via `count` |
| `db-read <domain> <table> […]` | Same via `read` (limit capped at 100) |
| `db-exists <domain> <table> --where EXPR` | Same via `exists` |
| `db-schema <domain> <table>` | Same via `schema` |
| `db-all-match <domain> <table> --condition EXPR` | Same via `all-match` |
| `db-unique <domain> <table> --columns a,b` | Same via `unique` |

`db-*` args are validated locally (domain/table shape, no `;`, no write SQL keywords) before building a POSIX-quoted remote argv. Never invent extra remote shell.

Successful `db-*` responses put parsed JSON under **`result`**.

## Mapping user questions → commands

| User question | Command |
|---------------|---------|
| "Can we reach the VPS?" | `ping` |
| "What host?" | `info` |
| "Pod status?" | `pods` |
| "Why is pod X failing?" | `describe-pod X`, then `logs X`; for structured JSON history see **chess-teacher-logs** |
| "Recent cluster events?" | `events` |
| "Is streamlit deployed?" | `deployments`, `rollout-status` |
| "Ingress / URL routing?" | `ingress`, `services` |
| "Disk / memory on VPS?" | `disk`, `memory` |
| "Prod DB row count / schema / uniqueness?" | `db-count` / `db-schema` / `db-unique` / … |

## Architecture (context)

- **Hetzner VPS** runs k3s/k8s; manifests under `/opt/chess_teacher/k8s/` (copied by CD).
- **CD** (push to `main`): build image → SCP manifests → SSH → `doppler run --config prod -- apply.sh`.
- **Namespace:** `chess-teacher`. Main workload: `deployment/streamlit`.
- **Prod Postgres** is firewalled from the laptop; `db-*` reaches it from inside the cluster via the streamlit pod’s injected `POSTGRES_*` env.

## Deploy note for `db-*`

Remote script: `scripts/tools/agent_db_query.py` (copied into the image). If `db-*` fails with file-not-found, merge/deploy this branch to main first, then retry.

## Troubleshooting

| Issue | Action |
|-------|--------|
| Doppler / ci secret missing | `doppler login`; verify access to `chess-teacher` / `ci` |
| SSH permission denied | Check `DEPLOY_SSH_KEY` in Doppler `ci`; key must match VPS `authorized_keys` |
| Windows `UNPROTECTED PRIVATE KEY FILE` / `CodexSandboxUsers` | Fixed in `vps_query.py`: key goes to `%LOCALAPPDATA%\chess-teacher-agent\`, not `%TEMP%`, with `icacls` lockdown. Update skill script if you see this on an old copy. |
| Windows `Load key ... invalid format` | Key file must use LF newlines only; CRLF from `NamedTemporaryFile` breaks OpenSSH. `vps_query.py` writes with `newline="\\n"`. |
| `ssh` not found | Install OpenSSH client (Windows optional feature) |
| Empty pods list | Wrong namespace or cluster issue — try `cluster-info`, `nodes` |
| Log command fails | Pod name from `pods` output; check `--container` if multi-container |
| `db-*` script missing | Image lag — wait for CD after merge, or confirm streamlit rolled to new tag |
| `db-*` JSON parse_error | Check `stderr` / `stdout` in the payload; often pod crash or import error |

## Script Jobs (mutating ops — user-run, not this skill)

For backfill / baseline reset / catch-up training on prod, the user SSHs to the VPS (or uses kubectl locally) and runs:

```bash
export PIPELINE_JOB_IMAGE="$(kubectl get deploy streamlit -n chess-teacher \
  -o jsonpath='{.spec.template.spec.containers[0].image}')"
python scripts/utils/run_script_job.py backfill_candidate_evals -- --workers 4
kubectl logs -n chess-teacher job/script-backfill-candidate-evals-YYYYMMDDHHMMSS -f
```

Whitelisted entrypoints only (`scripts/utils/run_script_job.py`). Template: `orchestration/k8s/job/script.yaml`. Dry-run: add `--dry-run` before `--`. Does not mount `chess-teacher-streamlit-secrets`.

| Task | Command |
|------|---------|
| Backfill candidate evals | `run_script_job.py backfill_candidate_evals -- --workers 4` |
| Reset baseline training | `run_script_job.py baseline_reset_training -- --yes` |
| Train until caught up | `run_script_job.py baseline_train_until_caught_up` |

Use `db-count` (read-only) here to verify backfill progress; never exec the backfill script into streamlit.

## References

Deploy script: `orchestration/k8s/apply.sh`. CD workflow: `.github/workflows/cd.yml`. Shared DB CLI: `scripts/tools/agent_db_query.py`. Script Jobs CLI: `scripts/utils/run_script_job.py`.
