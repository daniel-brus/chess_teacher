---
name: chess-teacher-logs
description: >-
  Read-only log investigation for chess_teacher: local JSON-lines buffer
  (active + pending .ready segments) and shipped S3 segments under
  logs/python/buffer. Search, tail, buffer status, segment listing. Routes
  questions to local vs S3 vs VPS kubectl logs. Use when debugging errors,
  tracing requests, checking log shipping, or finding historical prod logs.
---

# Chess Teacher Logs (log detective, read-only)

## How logging works (two tiers)

```
App log call
  → console (human-readable, ephemeral)
  → local buffer: LOG_BUFFER_DIR/active/app.log  (JSON-lines)
       ↓ rotate every 10 min or 5 MB
  → closed/YYYY/MM/DD/{HOSTNAME}/app-HHMMSSZ.log.ready
       ↓ LogShipper (~60s) when LOG_SHIP_ENABLED
  → S3 key: logs/python/buffer/closed/YYYY/MM/DD/{HOSTNAME}/app-HHMMSSZ.log
```

| Tier | Location | Best for |
|------|----------|----------|
| **Active buffer** | `{LOG_BUFFER_DIR}/active/*.log` | Last few minutes, current local/k8s pod run |
| **Pending closed** | `{LOG_BUFFER_DIR}/closed/**/*.ready` | Rotated but not yet uploaded (ship lag or failure) |
| **Shipped segments** | S3 prefix `logs/python/buffer/closed/...` | Historical search, prod after upload |
| **Container stdout** | `kubectl logs` on VPS | Live prod process output (not structured buffer) — use **chess-teacher-vps** |

Each JSON line has: `ts`, `level`, `logger`, `msg`, `log_id`, `environment`, optional `exc_type` / `exc_msg` / `traceback`.

Implementation: `src/chess_teacher/utils/logging/`.

## Doppler environments

| Config | Typical `LOG_BUFFER_DIR` | `LOG_SHIP_ENABLED` | Where logs land |
|--------|--------------------------|--------------------|-----------------|
| `dev_local` | `./storage/log-buffer` | often `false` locally | Mostly local buffer; S3 = MinIO if shipping on |
| `dev_k3d` | from secret | varies | MinIO via k3d endpoint |
| `prod` | e.g. `/tmp/chess-teacher-logs` in container | `true` | S3 (external); buffer is ephemeral inside pod |

Wrap commands with the config matching the environment under investigation:

```bash
doppler run --project chess-teacher --config dev_local -- python .agents/skills/chess-teacher-logs/scripts/log_query.py --json buffer-status
doppler run --project chess-teacher --config prod -- python .agents/skills/chess-teacher-logs/scripts/log_query.py --json search --source s3 --level ERROR --since 2h --limit 20
```

For **live prod pod stdout** (not buffered JSON files), use **chess-teacher-vps** `logs` — complementary, not a replacement.

## Routing: user question → first step

| Question | Start here | Why |
|----------|------------|-----|
| "What just happened locally?" | `buffer-status` → `tail` | Active file has freshest lines |
| "Any errors in the last hour (local)?" | `search --source local --level ERROR --since 1h` | Buffer may still hold unshipped segments |
| "Are logs shipping?" | `buffer-status` | `pending_upload_count` > 0 with ship enabled → lag or upload failure |
| "Prod errors yesterday" | `search --source s3 --level ERROR --date 2026/07/13 --environment PROD` | Shipped segments in S3 |
| "Logs from streamlit pod X" | `search --source s3 --hostname <pod-name>` or VPS `logs <pod>` | HOSTNAME = k8s pod name in segment path |
| "What log files exist for a date?" | `segments --date 2026/07/14` | Inventory before deep read |
| "Is S3 reachable for logs?" | chess-teacher-storage `list logs/python/buffer` | Raw key listing without parsing |

**Decision rule:** prefer **`local`** when the process is (or was recently) running on your machine or you care about the last ~10 minutes. Prefer **`s3`** for prod history or after local buffer was cleared. Use **`both`** when unsure.

## Who runs what

- **The user asks in chat** (e.g. "why did ingestion fail at 3pm?").
- **The agent runs** `.agents/skills/chess-teacher-logs/scripts/log_query.py`, interprets JSON, summarizes.
- Script sets **`ENVIRONMENT=AGENT`**. Read-only — never delete `.ready` files, trigger uploads, or modify buffer.

## Rules

- **Read-only only.** Local file reads + S3 `read_bytes` / `list_keys`. No writes, no `LogShipper.scan_once`, no deleting segments.
- Always **`--json`**. Repository root, `.venv` activated. No `uv`.
- `pytest` / `mypy` / `ruff` are allowed via the python-environment rule (venv), but not as part of this inspection skill unless the user asked.
- Summarize log matches; avoid dumping huge tracebacks unless asked.

## Agent workflow

1. Classify question using routing table above.
2. `info` or `buffer-status` if environment unclear.
3. Run targeted command; cross-use **storage** (key listing) or **vps** (live kubectl logs) if needed.

```bash
doppler run --project chess-teacher --config dev_local -- python .agents/skills/chess-teacher-logs/scripts/log_query.py --json buffer-status
doppler run --project chess-teacher --config dev_local -- python .agents/skills/chess-teacher-logs/scripts/log_query.py --json tail --lines 30
doppler run --project chess-teacher --config dev_local -- python .agents/skills/chess-teacher-logs/scripts/log_query.py --json search --source local --contains ingestion --level ERROR --since 2h
doppler run --project chess-teacher --config prod -- python .agents/skills/chess-teacher-logs/scripts/log_query.py --json search --source s3 --level ERROR --since 24h --limit 15
doppler run --project chess-teacher --config prod -- python .agents/skills/chess-teacher-logs/scripts/log_query.py --json segments --date 2026/07/14 --source both
```

## Script commands

| Command | Purpose |
|---------|---------|
| `info` | Layout, env flags, JSON fields, routing hint |
| `buffer-status` | Active log files, pending `.ready` count, recent pending list |
| `tail [--lines N]` | Last N JSON lines from local active file(s) |
| `search` | Filter across local and/or S3 (`--level`, `--logger`, `--contains`, `--since`, `--date`, `--hostname`) |
| `segments` | List local pending + S3 shipped segments |
| `read-segment <target>` | Read one file (relative under buffer dir or full S3 key) |

`--since`: ISO time or `30m`, `2h`, `1d`.
`--date`: path segment `YYYY/MM/DD` under `closed/`.
`--hostname`: pod name folder in segment paths (from k8s `metadata.name`).

## Related skills

| Skill | When |
|-------|------|
| **chess-teacher-storage** | List/count raw keys under `logs/python/buffer` without parsing JSON |
| **chess-teacher-vps** | Live prod `kubectl logs`, pod status (read-only) |
| **chess-teacher-db** | Correlate log timestamps with DB row state |

## Troubleshooting

| Issue | Action |
|-------|--------|
| Empty `tail` / `search --source local` | App not running or different `LOG_BUFFER_DIR`; check `info` |
| High `pending_upload_count` | Upload failures — search local pending segments; check storage connectivity with storage skill `health` |
| Empty S3 search | Wrong date/hostname; try `segments --source s3`; confirm `LOG_SHIP_ENABLED=true` in prod |
| `HOSTNAME` missing locally | Set in env for segment paths; k8s sets from pod name |
| Parse errors in matches | Non-JSON line in segment — shown as `PARSE_ERROR` level |

## API reference

Buffer/shipping: `src/chess_teacher/utils/logging/buffer.py`, `shipping.py`, `runtime.py`.
S3 key mapping: `log_storage_key_for_segment()`.
