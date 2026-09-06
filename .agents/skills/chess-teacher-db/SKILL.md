---
name: chess-teacher-db
description: >-
  Read-only Postgres inspection for chess_teacher using DatabaseClient and
  metadata.yml table keys. Runs validation queries (row counts, all-rows-match
  conditions, column uniqueness, schema diff). Use when the user asks about
  database contents, row counts, data quality, uniqueness, whether values satisfy
  a condition, schema drift, or postgres tables in this project. For production
  (firewalled VPS Postgres), use chess-teacher-vps db-* commands instead of
  doppler --config prod from the laptop.
---

# Chess Teacher Database (read-only)

## Doppler environments

Secrets live in Doppler project **`chess-teacher`**, not in git. Wrap skill commands with the config that matches the target Postgres:

| Config | `POSTGRES_HOST` (typical) | When to use |
|--------|---------------------------|-------------|
| `dev_local` | `localhost` (Compose) | Default local dev (`make dev_infra`) and k3d staging (`make dev_k3d_up` applies host overrides) |
| `prod` | External / VPS | **Prefer chess-teacher-vps `db-*`** — laptop often cannot reach firewalled prod Postgres |

Local Compose stack: Postgres, MinIO, and Redis from `docker-compose.infra.yml` with **`dev_local`**. Production Postgres is typically only reachable from the VPS; do **not** expect `doppler run --config prod` from the laptop to work. Use **chess-teacher-vps** `db-count` / `db-read` / … instead.

```bash
doppler run --project chess-teacher --config dev_local -- python scripts/tools/agent_db_query.py --json list-domains
```

Canonical script: **`scripts/tools/agent_db_query.py`**. The skill path `.agents/skills/chess-teacher-db/scripts/db_query.py` is a launcher to the same file. Prod uses the VPS skill (`db-*`), which kubectl-execs that script in the streamlit container.

The script loads `.env` when present; prefer **`doppler run --config … --`** so credentials match the intended environment.

## Who runs what

- **The user asks questions in chat** (e.g. “are columns Y and Z unique?”). They do **not** need to run the script or remember commands.
- **The agent runs** `scripts/tools/agent_db_query.py` (or the skill launcher) via the terminal for dev, or **chess-teacher-vps** `db-*` for prod; interprets `--json` output, and answers in plain language.
- The script sets **`ENVIRONMENT=AGENT`** before any `chess_teacher` import (overrides `LOCAL` from `.env` for that process). Do not ask the user to set this.

## Rules

- **Read-only only.** Use this skill's script or `DatabaseClient` read/introspection methods. Never call `insert`, `merge`, `overwrite`, `update_where`, `delete_where`, `truncate_table`, `drop_table`, or `ensure_metadata`.
- **Never** run raw `psql` or ad-hoc write SQL.
- Always pass **`--json`** when running the script.
- Run from **repository root** with `.venv` activated. Do not use `uv`.
- `pytest` / `mypy` / `ruff` are allowed via the python-environment rule (venv), but not as part of this inspection skill unless the user asked.

## Agent workflow

1. If target is **prod** → switch to **chess-teacher-vps** `db-*` (see that skill).
2. If domain or table is unknown → `list-domains`, then `list-tables <domain>`.
3. Pick the command that matches the question (see table below).
4. Run the script; summarize results for the user.

```bash
doppler run --project chess-teacher --config dev_local -- python scripts/tools/agent_db_query.py --json list-domains
doppler run --project chess-teacher --config dev_local -- python scripts/tools/agent_db_query.py --json list-tables pipelines/ingestion
doppler run --project chess-teacher --config dev_local -- python scripts/tools/agent_db_query.py --json unique pipelines/ingestion raw_games --columns account_id,platform_game_id
```

On Windows (PowerShell), same commands with `.venv\Scripts\python.exe` if the venv is not activated.

## Domains (discovered, not hardcoded)

A **domain** is the folder path (under the installed `chess_teacher` package) that contains a `metadata.yml` with a `tables:` section — e.g. `ingestion`, `pipelines`, `other`.

- Discovered at runtime by scanning the package tree (`list-domains`).
- New modules only need a folder + `metadata.yml`; no skill or script edits.
- `table` arguments are **YAML keys** under `tables:` (e.g. `raw_games`), not always the SQL table name.

## Script commands

`python scripts/tools/agent_db_query.py --json <command> ...`

| Command | Purpose |
|---------|---------|
| `list-domains` | Discover all domains + `metadata_path` |
| `list-tables <domain>` | List table keys in that domain |
| `read <domain> <table>` | Sample rows (`--where`, `--columns`, `--order-by`, `--limit`) |
| `count <domain> <table>` | Row count (`--where` optional) |
| `exists <domain> <table> --where EXPR` | Any row matches? |
| `schema <domain> <table>` | `table_exists`, `schema_diff` summary |
| `all-match <domain> <table> --condition EXPR` | Every row satisfies boolean EXPR? |
| `unique <domain> <table> --columns col1,col2` | Combination unique? |

`EXPR` / `--condition` / `--where`: SQL boolean expressions **without** the `WHERE` keyword. Column names must exist in metadata. Semicolons and write keywords are rejected.

## Mapping user questions → commands

| User question | Command |
|---------------|---------|
| "How many rows in X?" | `count` |
| "Show me some rows where …" | `read --where "…"` |
| "Does any row …?" | `exists --where "…"` |
| "Does **every** row satisfy …?" | `all-match --condition "…"` |
| "Are columns Y and Z unique together?" | `unique --columns Y,Z` |
| "Is column Y unique?" | `unique --columns Y` |
| "Does the table exist / match metadata?" | `schema` |

Interpret `all-match`: `all_match: true` and `violations: 0`. Use `sample_violations` when false.

Interpret `unique`: `is_unique: true` and `duplicate_group_count: 0`. Use `sample_duplicate_groups` when false.

## In-process alternative

Prefer the script. If needed, set `os.environ["ENVIRONMENT"] = "AGENT"` **before** importing `chess_teacher`, then use read-only `DatabaseClient` APIs. Resolve `yaml_path` via `list-domains` output, not hardcoded paths.

## Troubleshooting

| Issue | Action |
|-------|--------|
| `ModuleNotFoundError: chess_teacher` | Activate `.venv`; `pip install -r requirements-dev.txt` |
| Connection errors (dev) | Postgres up; use `doppler run --config dev_local` |
| Connection timeout (prod from laptop) | Expected if firewalled — use **chess-teacher-vps** `db-*` |
| Unknown domain / table | `list-domains` / `list-tables` |
| Invalid column | Check that domain's `metadata.yml` |

## API reference

Read/introspection methods only: `src/chess_teacher/utils/db/client.py`.
