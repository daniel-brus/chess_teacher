---
name: chess-teacher-redis
description: >-
  Read-only Redis cache inspection for chess_teacher using REDIS_URL and
  redis-py. PING, SCAN keys, TTL/size metadata for user cache keys. Never
  writes or deletes. Use when the user asks about Redis connectivity, cache
  keys, TTLs, cache hits/misses debugging, or whether user games/accounts
  are cached.
---

# Chess Teacher Redis (read-only)

## Doppler environments

Secrets live in Doppler project **`chess-teacher`**, not in git. Wrap skill commands with the config that matches the target backend:

| Config | `REDIS_URL` target | When to use |
|--------|-------------------|-------------|
| `dev_local` | `redis://localhost:6379/0` (Compose) | Local dev and k3d staging (pods use `host.k3d.internal` via `apply-k3d-local.ps1`) |
| `prod` | External managed Redis (e.g. Upstash) | **Production cache — read-only only** |

Local Compose stack: Postgres, MinIO, and Redis from `docker-compose.infra.yml` with **`dev_local`**. Production uses external Postgres, S3, and Redis from **`prod`**.

```bash
doppler run --project chess-teacher --config dev_local -- python .agents/skills/chess-teacher-redis/scripts/redis_query.py --json ping
```

Override config when inspecting prod: `--config prod` (extra care — shared production cache).

## Who runs what

- **The user asks questions in chat** (e.g. “is Redis up?” or “any keys for user X?”).
- **The agent runs** `.agents/skills/chess-teacher-redis/scripts/redis_query.py` via the terminal, interprets JSON output, and answers in plain language.
- The script sets **`ENVIRONMENT=AGENT`** before loading env. Prefer **`doppler run --config … --`** so `REDIS_URL` is injected; `.env` works if populated locally.

## Rules

- **Read-only only.** Allowed: `PING`, `INFO` (memory section), `DBSIZE`, `SCAN`, `EXISTS`, `TTL`, `TYPE`, `STRLEN`. **Never** `SET`, `DEL`, `FLUSHDB`, `FLUSHALL`, `CONFIG SET`, `DEBUG`, `SHUTDOWN`, `KEYS` (use `scan`), or raw `redis-cli` write commands.
- **Never** call `get_cache_client()` write paths (`set_*`, `delete`, invalidation helpers) from this skill.
- Always pass **`--json`**. Run from **repository root** with `.venv` activated. Do not use `uv`.
- Do not run `pytest` / `mypy`; ask the user to run those manually.
- **`--show-value`** is opt-in and only decodes small JSON for `*:accounts:v1` keys; games keys are Parquet blobs — use `key-info` size/TTL only.

## App cache keys

From `src/chess_teacher/utils/cache_utils.py`:

| Key pattern | Content | TTL |
|-------------|---------|-----|
| `user:{user_id}:games:v1` | Parquet-encoded Polars DataFrame | 3600s |
| `user:{user_id}:accounts:v1` | JSON list of accounts | 1800s |

## Agent workflow

1. Unknown target → `info`, then `ping`.
2. Pick command (table below).
3. Run via Doppler with the correct config; summarize for the user.

```bash
doppler run --project chess-teacher --config dev_local -- python .agents/skills/chess-teacher-redis/scripts/redis_query.py --json ping
doppler run --project chess-teacher --config dev_local -- python .agents/skills/chess-teacher-redis/scripts/redis_query.py --json scan --pattern "user:*" --limit 20
doppler run --project chess-teacher --config dev_local -- python .agents/skills/chess-teacher-redis/scripts/redis_query.py --json key-info user:abc123:accounts:v1
```

On Windows (PowerShell), same commands with `.venv\Scripts\python.exe` if the venv is not activated.

## Script commands

`doppler run --project chess-teacher --config <config> -- python .agents/skills/chess-teacher-redis/scripts/redis_query.py --json <command> ...`

| Command | Purpose |
|---------|---------|
| `info` | Endpoint label, DB index, cache key patterns (no network) |
| `ping` | Connectivity check |
| `dbsize` | Key count in DB |
| `memory` | Memory usage summary from `INFO memory` |
| `scan [--pattern P] [--limit N]` | List keys (default pattern `user:*`) |
| `exists <key>` | Key present? |
| `key-info <key>` | Type, TTL, size; optional `--show-value` for accounts JSON |

## Mapping user questions → commands

| User question | Command |
|---------------|---------|
| "Is Redis reachable?" | `ping` |
| "Which Redis / DB?" | `info` |
| "How many cache keys?" | `dbsize` or `scan user:*` |
| "Keys for a user?" | `exists user:{id}:games:v1` and `exists user:{id}:accounts:v1` |
| "TTL / size of a key?" | `key-info <key>` |
| "Memory usage?" | `memory` |

## Troubleshooting

| Issue | Action |
|-------|--------|
| `REDIS_URL is not set` | Run with `doppler run --config dev_local` or start `make dev_infra` |
| Connection refused on `dev_local` | `make dev_infra`; Redis service in `docker-compose.infra.yml` |
| Empty `scan` | Cache miss or different DB; confirm `REDIS_URL` path (`/0`) |
| `ModuleNotFoundError: redis` | Activate `.venv`; `pip install -r requirements-dev.txt` |

## API reference

Cache client (do not use write methods from this skill): `src/chess_teacher/utils/cache_utils.py`.
