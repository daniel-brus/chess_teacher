# VPS source (enabled for Triage)

Triage scans live kubectl logs via **chess-teacher-vps** when `DEPLOY_*` are available.

## Credentials

`prod_agent_triage` includes `DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_SSH_KEY` (copied from `ci`).

Cloud Agent: `DOPPLER_TOKEN_TRIAGE` → `doppler run --config prod_agent_triage` injects SSH + S3.

`vps_query.py` prefers **env** `DEPLOY_*`; falls back to Doppler `ci` for local interactive use.

## Commands Triage should use

```bash
DOPPLER_TOKEN="$DOPPLER_TOKEN_TRIAGE" doppler run --project chess-teacher --config prod_agent_triage -- \
  .venv/bin/python .agents/skills/chess-teacher-vps/scripts/vps_query.py --json ping

DOPPLER_TOKEN="$DOPPLER_TOKEN_TRIAGE" doppler run --project chess-teacher --config prod_agent_triage -- \
  .venv/bin/python .agents/skills/chess-teacher-vps/scripts/vps_query.py --json image-tag

DOPPLER_TOKEN="$DOPPLER_TOKEN_TRIAGE" doppler run --project chess-teacher --config prod_agent_triage -- \
  .venv/bin/python .agents/skills/chess-teacher-vps/scripts/vps_query.py --json pods

# Then logs for ERROR/crash pods (tail capped by TRIAGE_MAX_LOG_LINES spirit):
DOPPLER_TOKEN="$DOPPLER_TOKEN_TRIAGE" doppler run --project chess-teacher --config prod_agent_triage -- \
  .venv/bin/python .agents/skills/chess-teacher-vps/scripts/vps_query.py --json logs <pod> --tail 200
```

## Rules

- Lock `--source vps` before scanning; release after.
- On SSH/`ping` failure: `source_ok=false`, **do not** advance VPS watermark; continue S3/GHA.
- Use `image-tag` for `provenance.observed_image_tag` / `observed_ref`.
- Issues from VPS: label `from-vps` (+ `agent-fix` when acting).
- Read-only only — never apply/restart via this path.
