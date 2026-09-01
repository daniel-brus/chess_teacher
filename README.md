# Chess Teacher

Streamlit chess teaching app with local dev infra (Postgres, MinIO, Redis via Docker Compose) and production deploy on Hetzner k8s. Secrets are managed in [Doppler](https://www.doppler.com/).

## Features

- Interactive chess board visualization with Streamlit
- Move history and game statistics
- Stockfish integration for analysis
- Pipelines for game ingestion and preprocessing
- Local dev stack or cloud backends (Supabase, S3, Redis) via Doppler configs

## Prerequisites

- Python 3.12
- Docker Desktop (Compose + optional k3d)
- [Doppler CLI](https://docs.doppler.com/docs/install-cli) (`winget install doppler.doppler`)
- Stockfish (for local venv runs; included in Docker image)

## Secrets (Doppler)

| Config | Purpose |
|--------|---------|
| `dev_local` | Local venv + Compose (`localhost` hosts) + k3d staging (`make dev_k3d_up`) |
| `ci` | GitHub Actions (Docker Hub, deploy SSH) |
| `prod` | Production VPS / cloud sync source |

Copy keys from [`.env.example`](.env.example) into Doppler `dev_local`.

```powershell
doppler login
# Set secrets in dashboard for config dev_local
```

## Local development

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .

# Start local Postgres + MinIO + Redis
make dev_infra

# Run Streamlit (venv)
make streamlit_fg

# Or Streamlit in Docker
make streamlit_docker
```

Streamlit opens at `http://localhost:<APP_PORT>` (default `8502` from Doppler `dev_local`).

### Local k3d staging (prod-like)

Runs Streamlit + ingestion CronJobs + pipeline Jobs against local Compose infra, using the **`develop` Docker image**. See [orchestration/k8s/README.md](orchestration/k8s/README.md).

```powershell
make streamlit_k3d    # infra + k3d + apply + port-forward → http://localhost:8501
```

Uses **`dev_local` Doppler only** — k3d host overrides are applied automatically (no second config).

### Cloud → local data clone

Requires Docker Desktop and AWS CLI on PATH (`pg_dump`/`pg_restore` run in a `postgres:17` container to match Supabase).

```powershell
make dev_sync_cloud
```

### Empty local DB schema only

```powershell
make dev_bootstrap_schema
```

## Makefile targets

| Target | Description |
|--------|-------------|
| `dev_infra` | Start Postgres, MinIO, Redis (no health wait) |
| `dev_down` | Stop Compose stack |
| `dev_bootstrap` | `dev_infra` + wait until healthy |
| `streamlit_fg` | Streamlit in venv (Doppler `dev_local`) |
| `streamlit_docker` | Streamlit container + infra |
| `streamlit_k3d` | Infra + k3d staging + port-forward (`:8501`) |
| `dev_k3d_up` | Same deploy as `streamlit_k3d` without port-forward |
| `dev_sync_cloud` | Full prod → local Postgres + MinIO sync |
| `dev_bootstrap_schema` | `ensure_metadata` for all tables |

Override Doppler config: `make streamlit_fg DOPPLER_CONFIG_LOCAL=prod` (uses cloud backends).

## Tests and linting

Run manually in your terminal:

```powershell
pytest
ruff check src scripts tests
mypy src
```

## Production deploy

Push to `main` triggers CD: build Docker image, SCP k8s manifests, `doppler run --config prod -- apply.sh` on VPS.

GitHub secrets: `DOPPLER_CICD_SERVICE_TOKEN` (repo), `DOPPLER_SERVICE_TOKEN` (production env), `VERSION_BUMP_PAT`.
