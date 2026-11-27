# VPS source (optional / v2) — enabled path documented, not wired in v1 Automations

Plan sources **v1 = S3 + GitHub Actions**. VPS kubectl logs are **optional**.

Triage Automation prompts currently exclude VPS. Empty `agent/triage/state/vps.json` is initialized so enabling later is a prompt + secrets change only.

## Options

1. **Cursor-hosted + SSH in `prod_agent_triage`**
   ```bash
   # copy from ci into prod_agent_triage when ready
   doppler secrets set DEPLOY_HOST=... DEPLOY_USER=... DEPLOY_SSH_KEY=... \
     --project chess-teacher --config prod_agent_triage
   ```
   Triage then uses `chess-teacher-vps` / `vps_query.py logs`.

2. **Self-hosted Cursor worker** (preferred if SSH keys must not leave your net)
   Keep SSH on Doppler `ci` only on that machine; run Triage with self-hosted runtime for VPS scans.

## Enable checklist

- [ ] Choose option 1 or 2
- [ ] Confirm `state-load --source vps` works
- [ ] Extend Triage Automation prompt: lock/scan VPS since watermark; label `from-vps`
- [ ] On SSH failure: `source_ok=false`, do not advance VPS watermark, continue S3/GHA

## Labels

`from-vps` (created by `ensure_labels.ps1`).
