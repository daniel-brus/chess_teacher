---
name: import-dag
description: >
  Package and import layering for chess_teacher as a DAG (folders + files).
  Use when adding modules, moving code, fixing circular imports, choosing a
  folder for new code, wrapping third-party imports, or reviewing whether an
  import points the wrong way. Auto-trigger on structural refactors and new
  cross-package imports.
---

# Import DAG

Package `chess_teacher` must stay a **directed acyclic graph** of dependencies.
Edges point from consumer → dependency (importer depends on importee).
Cycles forbidden at folder level and file level.

Lazy / `TYPE_CHECKING` imports may avoid runtime crashes but do **not** count as a layering fix unless the dependency truly belongs at a higher tier (example: log shipping may lazy-import object storage because shipping sits above core logging). Prefer restructuring over hiding cycles.

## Top-level layers (low → high)

Lower may not import higher. Same-layer: prefer depending downward.

1. `utils` — shared primitives only (see utils sub-DAG)
2. `platform` — accounts/users; may use `utils` only
3. `pipelines` — ingestion / preprocessing / neural_network; may use `utils`, `platform`
4. `bots` — interactive opponents; may use `utils`, `platform`, `pipelines` (including NN)
5. Orchestration / one-off: `maintenance`, `backfill`, thin CLIs — may use lower layers; must not be imported by `utils`, `platform`, or `bots`

Avoid new code in `other/`. Place by layer instead (domain type → `platform` / `pipelines` / `utils`; script → `maintenance` / `backfill`).

**Hard rule:** `utils` never imports `platform`, `pipelines`, `bots`, `maintenance`, `backfill`, or `other`.

## Utils sub-DAG (low → high)

1. `env_utils`, `exception_utils`, `general_utils` — no `chess_teacher` deps (stdlib / third-party only)
2. `process_utils` — process helpers; may use layer 1; must not import the logging package root if that pulls shipping
3. **Logging core** — `logging.logger`, `logging.buffer`, `logging.formatters`, `logging.config`, `logging.runtime` (without shipping at import time). Mid-layer utils import `get_logger` / `EnhancedLogger` from these modules, not via patterns that force shipping
4. `files`, `metadata_utils`, `db`, `chess_utils`, `cache_utils` (cache stores primitives / dicts / polars — not platform types)
5. `object_storage`, then **log shipping** (`logging.shipping`) — shipping may use object storage; core logging must not import shipping at module load
6. `pipeline_utils`, `table_data_class`

Within a utils package, files also form a DAG: leaf helpers first; fat `__init__` re-exports last (prefer lazy / empty package `__init__`). Prefer narrow imports (`logging.config`, `logging.logger`) over the package root when that preserves the DAG.

## Third-party imports

Be sceptical of non-`chess_teacher` imports in feature code:

- Stdlib / config reused twice+ → wrap in `general_utils` or `env_utils` (e.g. `get_current_datetime`, `get_env_variable`, `get_optional_env_variable`, `load_yaml`)
- Do not call `os.getenv` / raw dotenv outside `env_utils`
- Heavy libs (tensorflow, mlflow, boto3, redis): import next to the owning module; add `*_utils` only if the same boilerplate repeats across modules
- Domain libs used as vocabulary (`chess`, `polars`) may import directly in domain code; shared patterns go to the right utils package (`chess_utils`, dataframe helpers in `pipeline_utils`)

## Where new code goes

| Kind | Put in |
|---|---|
| Env / config / datetime / hash / yaml / SQL-ident helpers | `utils/env_utils` or `utils/general_utils` |
| Logging primitives | `utils/logging` core modules (no storage imports) |
| Log shipping to storage | `utils/logging/shipping.py` (above object storage) |
| Storage I/O | `utils/object_storage` / `utils/files` |
| DB access | `utils/db` |
| Pipeline step base / transforms plumbing | `utils/pipeline_utils` |
| Account / user identity | `platform` |
| Batch transforms on games / moves / models | matching `pipelines/*` |
| NN train / load / encode | `pipelines/neural_network` (not `utils`) |
| Interactive chess bots (incl. NN policy bots) | `bots` |

If unsure: place at the **highest** layer that still keeps the DAG (prefer feature folder over `utils`). Wrong folder in `utils` is worse than wrong feature subfolder.

## Agent checklist before adding an import

1. Does this edge point **down** the layer list?
2. Would this create / worsen a cycle (esp. logging core ↔ storage ↔ files)?
3. Is this third-party call a one-off, or should it be a named helper in utils?
4. Am I dumping into `other/` because unsure? Stop — pick a layer.

## Known remaining debt

- `other/` still exists as a catch-all (freeze: no new files; relocate over time)
- `other` → `pipelines` one-way edges remain (e.g. openings slug scan uses `Game` metadata) — prefer moving openings fully under preprocessing later
- Module-level `get_logger()` in some mid/high modules can still pull shipping early via `configure_logging` — prefer lazy loggers when adding new utils
