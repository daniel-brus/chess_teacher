---
name: chess-teacher-storage
description: >-
  Read-only raw object storage inspection for chess_teacher using ObjectStorage
  and get_raw_storage(). Lists keys by prefix, checks object/prefix existence,
  counts files, and shows immediate "folder" children. Works with S3 and local
  S3-compatible backend from .env. Use when the user asks what files or folders
  exist under a path, whether a key exists, storage layout, ingested/processed
  prefixes, or S3/object storage contents in this project.
---

# Chess Teacher Storage (read-only)

## Doppler environments

Secrets live in Doppler project **`chess-teacher`**, not in git. Wrap skill commands with the config that matches the target object storage:

| Config | S3 endpoint (typical) | When to use |
|--------|----------------------|-------------|
| `dev_local` | `http://localhost:9000` (MinIO Compose) | Local dev and k3d staging (pods use `host.k3d.internal:9000` via `apply-k3d-local.ps1`) |
| `prod` | External S3 / Supabase storage | **Production objects — read-only only** |

Local Compose stack uses MinIO with **`dev_local`**. Production uses external S3 from **`prod`** (source for `make dev_sync_cloud` object copy).

```bash
doppler run --project chess-teacher --config dev_local -- python .agents/skills/chess-teacher-storage/scripts/storage_query.py --json info
```

The script loads `.env` when present; prefer **`doppler run --config … --`** so `S3_*` and `STORAGE_ROOT` match the intended environment.

## Prefixes, not folders

Object storage has **keys**, not directories. Keys look like POSIX paths (`ingested/acct/2026/06/file.jsonl`) relative to `STORAGE_ROOT`. A **prefix** is the same idea as a folder path: listing under `ingested/acct` means all keys that start with that prefix. The `children` command derives immediate sub-prefix names from deeper keys.

## Who runs what

- **The user asks questions in chat** (e.g. “what’s under `logs/python`?” or “does `ingested/foo/bar.jsonl` exist?”).
- **The agent runs** `.agents/skills/chess-teacher-storage/scripts/storage_query.py` via the terminal, interprets JSON output, and answers in plain language.
- The script sets **`ENVIRONMENT=AGENT`** before any `chess_teacher` import. Do not ask the user to set this.

## Rules

- **Read-only only.** Use this skill's script or read-only `ObjectStorage` methods (`list_keys`, `read_bytes`). Never call `write_*`, `move`, `delete`, or `delete_keys`.
- Always pass **`--json`** when running the script (output is JSON either way; flag kept for parity with the DB skill).
- Run from **repository root** with `.venv` activated. Do not use `uv`.
- Keys are **relative to `STORAGE_ROOT`** (from `.env`), not the full S3 bucket key. The backend adds `STORAGE_ROOT` as a prefix in S3 automatically.
- Do not run `pytest` / `mypy`; ask the user to run those manually.

## Agent workflow

1. If backend or root is unknown → `info`.
2. Pick the command that matches the question (see table below).
3. Run the script; summarize results for the user.

```bash
doppler run --project chess-teacher --config dev_local -- python .agents/skills/chess-teacher-storage/scripts/storage_query.py --json info
doppler run --project chess-teacher --config dev_local -- python .agents/skills/chess-teacher-storage/scripts/storage_query.py --json children logs/python
doppler run --project chess-teacher --config dev_local -- python .agents/skills/chess-teacher-storage/scripts/storage_query.py --json list ingested/acct --suffix jsonl --limit 20
doppler run --project chess-teacher --config dev_local -- python .agents/skills/chess-teacher-storage/scripts/storage_query.py --json exists ingested/acct/file.jsonl
doppler run --project chess-teacher --config dev_local -- python .agents/skills/chess-teacher-storage/scripts/storage_query.py --json any-under processed/acct
```

On Windows (PowerShell), same commands with `.venv\Scripts\python.exe` if the venv is not activated.

## Script commands

`python .agents/skills/chess-teacher-storage/scripts/storage_query.py --json <command> ...`

| Command | Purpose |
|---------|---------|
| `info` | Backend (`s3`), `storage_root`, bucket, endpoint |
| `list [prefix]` | List object keys (`--no-recursive`, `--suffix`, `--glob`, `--limit`) |
| `children [prefix]` | Immediate sub-prefixes + direct files under prefix |
| `exists <key>` | Single object exists? (`read_bytes` is not None) |
| `any-under <prefix>` | Any objects under prefix? (use for “does this folder exist?”) |
| `count [prefix]` | Count keys under prefix |
| `match <pattern> [prefix]` | Shell glob via fnmatch on keys under prefix |
| `health` | Connectivity probe (writes/deletes a `_healthcheck` probe — use only when user asks) |

`--suffix`: e.g. `jsonl` or `.jsonl` (same as pipelines).
`--glob`: regex passed to `ObjectStorage.list_keys(glob_pattern=...)`.

## Mapping user questions → commands

| User question | Command |
|---------------|---------|
| "What's in / under X?" | `children X` (folders + files) or `list X` (all keys) |
| "List all `.jsonl` under ingested/foo" | `list ingested/foo --suffix jsonl` |
| "Does file X exist?" | `exists X` |
| "Does folder/prefix X exist?" | `any-under X` |
| "How many files under X?" | `count X` |
| "Which backend / bucket?" | `info` |
| "Is storage reachable?" | `health` |
| "List raw log segment keys" | `list logs/python/buffer` |
| "Search or parse log content" | Use **chess-teacher-logs** (storage lists keys only) |

Interpret `exists`: `exists: true` means the object is present.
Interpret `any-under`: `any: true` means at least one key starts with that prefix (empty prefix at root lists everything).
Interpret `children`: `folders` are immediate sub-prefixes; `files` are objects directly under the prefix (no `/` in the remainder).

## Common prefixes in this project

| Prefix | Typical contents |
|--------|------------------|
| `ingested/<account_id>/` | New JSONL awaiting ingestion |
| `processed/<account_id>/` | Successfully ingested JSONL |
| `failed/<account_id>/` | Failed ingestion JSONL |
| `logs/python/` | Shipped Python application logs (see **chess-teacher-logs** skill to search/parse JSON segments) |
| `assets/images/` | Platform asset images |

Account IDs and dates appear as further path segments.

## In-process alternative

Prefer the script. If needed, set `os.environ["ENVIRONMENT"] = "AGENT"` **before** importing `chess_teacher`, then:

```python
from chess_teacher.utils.object_storage.factory import get_raw_storage

storage = get_raw_storage()
keys = storage.list_keys("ingested/acct", recursive=True, suffix="jsonl")
exists = storage.read_bytes("path/to/key.jsonl") is not None
```

## Troubleshooting

| Issue | Action |
|-------|--------|
| `ModuleNotFoundError: chess_teacher` | Activate `.venv`; `pip install -r requirements-dev.txt` |
| S3 / connection errors | `doppler run --config dev_local`; check `S3_*`, `STORAGE_ROOT`; try `health` |
| Empty `list` / `any: false` | Prefix may be wrong (no leading slash); try `children` at parent prefix |
| `health` modifies storage | Creates/deletes `_healthcheck/<uuid>.txt` only; mention if user did not ask |

## API reference

`ObjectStorage.list_keys`, `read_bytes`: `src/chess_teacher/utils/object_storage/`.
Factory: `get_raw_storage()` in `src/chess_teacher/utils/object_storage/factory.py`.
