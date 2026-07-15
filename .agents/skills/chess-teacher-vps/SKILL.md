---
name: chess-teacher-vps
description: >-
  Read-only inspection of the production Hetzner VPS (k3s/k8s) via SSH.
  Whitelisted kubectl get/describe/logs, cluster-info, and host df/free/uptime
  only — never apply, delete, restart, or ad-hoc shell. SSH credentials from
  Doppler ci config. Use when the user asks about production pods, deploy
  status, ingress, logs, or VPS health.
---

# Chess Teacher VPS / Hetzner (read-only)

## ⚠️ Production safety

This is the **live production Hetzner VPS** running k8s (`chess-teacher` namespace). Treat every action as high-risk.

**The agent MUST:**

- Use **only** `.agents/skills/chess-teacher-vps/scripts/vps_query.py` subcommands — fixed, whitelisted remote commands.
- Run **read-only** operations: `kubectl get`, `kubectl describe`, `kubectl logs`, `kubectl cluster-info`, `kubectl rollout status` (status only, no restart).
- Prefer **`info`** and **`ping`** before deeper inspection.

**The agent MUST NOT:**

- Run arbitrary SSH commands, interactive shells, or pass user text as remote shell.
- Run `kubectl apply`, `delete`, `patch`, `scale`, `rollout restart`, `exec`, `port-forward`, or edit manifests on the VPS.
- Run `systemctl stop/restart`, `docker rm/stop`, `rm`, `reboot`, `doppler secrets set`, or anything that changes state.
- Fetch or print Doppler **secret values** from the VPS (only SSH connection metadata via `info`).
- Deploy, sync, or run `apply.sh` unless the user **explicitly** requests a deploy in a separate, deliberate step.

If the user needs a mutating change (restart pod, deploy, config edit), **stop and ask for explicit confirmation** — do not use this skill.

## Doppler environments

| Config | Role | Used by this skill? |
|--------|------|---------------------|
| `dev_local` | Local Compose (Postgres, MinIO, Redis on localhost) | No — use db/storage/redis skills |
| `dev_k3d` | Local k3d cluster | No — use kubectl locally |
| `ci` | GitHub Actions + **SSH to VPS** (`DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_SSH_KEY`) | **Yes** — script reads SSH from here |
| `prod` | Runtime secrets on VPS (Postgres, S3, Redis URLs for the app) | No for SSH; app secrets are injected on VPS at deploy time |

The VPS script loads SSH credentials from Doppler **`ci`** automatically. Requires `doppler login` with access to project `chess-teacher`.

Runtime app config on the cluster comes from **`prod`** via `doppler run` during CD (`orchestration/k8s/apply.sh`). Do not dump prod secrets through this skill.

## Who runs what

- **The user asks questions in chat** (e.g. “are streamlit pods healthy?” or “recent prod logs?”).
- **The agent runs** `vps_query.py` subcommands, interprets JSON (`stdout` / `stderr`), and answers in plain language.
- The script sets **`ENVIRONMENT=AGENT`**. SSH key is written to a temp file with `0600` and deleted immediately after use.

## Rules

- **Read-only only.** No exceptions without explicit user approval outside this skill.
- Always pass **`--json`**. Run from **repository root**. Do not use `uv`.
- Do not run `pytest` / `mypy`; ask the user to run those manually.
- Log output may contain user data — summarize; do not paste large log dumps unless asked.

## Agent workflow

1. `info` → confirm target host (no secrets).
2. `ping` → SSH works.
3. Pick command (pods, logs, events, …).

```bash
python .agents/skills/chess-teacher-vps/scripts/vps_query.py --json info
python .agents/skills/chess-teacher-vps/scripts/vps_query.py --json ping
python .agents/skills/chess-teacher-vps/scripts/vps_query.py --json pods
python .agents/skills/chess-teacher-vps/scripts/vps_query.py --json logs streamlit-xxxxx-yyyyy --tail 50
```

On Windows (PowerShell): `.venv\Scripts\python.exe` optional (script has no chess_teacher imports); needs `ssh` and `doppler` on PATH. The script writes the deploy key under `%LOCALAPPDATA%\chess-teacher-agent\` with LF newlines and locked-down ACLs — required because OpenSSH rejects keys in `%TEMP%` when Cursor/sandbox adds extra principals.

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

## Architecture (context)

- **Hetzner VPS** runs k3s/k8s; manifests under `/opt/chess_teacher/k8s/` (copied by CD).
- **CD** (push to `main`): build image → SCP manifests → SSH → `doppler run --config prod -- apply.sh`.
- **Namespace:** `chess-teacher`. Main workload: `deployment/streamlit`.

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

## References

Deploy script: `orchestration/k8s/apply.sh`. CD workflow: `.github/workflows/cd.yml`.
