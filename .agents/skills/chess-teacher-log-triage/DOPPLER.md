# Doppler config: `prod_agent_triage`

Created under Doppler project **`chess-teacher`**, environment **Production**.

> Naming: Doppler requires configs in `prod` to start with `prod_`. Conceptually this is the plan’s **`agent_triage`** config. Always pass `--config prod_agent_triage`.

## Included secrets

### Object storage (from prod S3 subset)

- `S3_BUCKET`
- `STORAGE_ROOT`
- `S3_ENDPOINT_URL`
- `S3_ACCESS_KEY_ID`
- `S3_SECRET_ACCESS_KEY`

Needed to read `logs/python/buffer/**` and read/write `agent/triage/**`.

### VPS SSH (from `ci`)

- `DEPLOY_HOST`
- `DEPLOY_USER`
- `DEPLOY_SSH_KEY`

Needed for chess-teacher-vps (`ping`, `image-tag`, kubectl logs). Prefer env injection via `doppler run`; `vps_query.py` falls back to Doppler `ci` locally.

### Triage knobs

| Secret | Default |
|--------|---------|
| `TRIAGE_OPEN_THRESHOLD` | `0.70` |
| `TRIAGE_DEBATE_BAND` | `0.15` |
| `TRIAGE_MAX_ACTS` | `5` |
| `TRIAGE_MAX_EVALS` | `30` |
| `TRIAGE_MAX_LOG_LINES` | `500` |
| `TRIAGE_GRACE_HOURS` | `6` |
| `TRIAGE_CLOSED_LOOKBACK_DAYS` | `60` |
| `TRIAGE_FLAP_REOPENS` | `3` |
| `TRIAGE_FLAP_DAYS` | `14` |
| `TRIAGE_LOCK_TTL_MINUTES` | `45` |

## Explicitly excluded

Postgres, Redis, Streamlit OAuth, DockerHub, `LOG_BUFFER_DIR`.

## Local usage

```bash
doppler run --project chess-teacher --config prod_agent_triage -- .venv/Scripts/python.exe .agents/skills/chess-teacher-log-triage/scripts/triage_query.py --json settings
```

## Cloud Automations (Cursor)

**Use a separate secret name** so general Cloud Agents keep their existing `DOPPLER_TOKEN`:

1. Doppler → Project `chess-teacher` → Access → Service Tokens → create token for config **`prod_agent_triage`** (read).
2. In the Cursor Cloud Agent **environment** secrets, set:
   - `DOPPLER_TOKEN_TRIAGE` = that service token
   (leave existing `DOPPLER_TOKEN` alone for other tasks)
3. Cloud Agent snapshot installs the Doppler CLI via `scripts/dev/cloud_agent_infra_install.sh`.
4. Automation prompts must map the name Doppler expects:

```bash
DOPPLER_TOKEN="$DOPPLER_TOKEN_TRIAGE" doppler run \
  --project chess-teacher --config prod_agent_triage -- \
  .venv/bin/python .agents/skills/chess-teacher-log-triage/scripts/triage_query.py --json …
```

Do **not** put full `prod` credentials in Cursor. Fallback (no Doppler CLI): paste `S3_*` + `TRIAGE_*` into Cursor env instead.

## Recreate / update

```bash
doppler configs create --project chess-teacher --environment prod --name prod_agent_triage
# copy S3_* from prod, set TRIAGE_*, delete unrelated secrets
```
