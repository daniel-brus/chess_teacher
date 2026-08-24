# Local k3d staging (prod-like)

Mimics production on your laptop: **Compose** for Postgres/MinIO/Redis, **k3d** for Streamlit + CronJobs + pipeline Jobs. Uses the **`develop` Docker image** (`{DOCKERHUB}/chess_teacher:develop`).

No separate Doppler config — **`dev_local` only**. Host routing overrides live in `apply-k3d-local.ps1`.

## Prerequisites

- Docker Desktop running
- `doppler login` + `dev_local` secrets populated (see repo `.env.example`)
- `k3d`, `kubectl` on PATH
- Google OAuth: add redirect URI `http://localhost:8501/oauth2callback` (k8s port-forward; venv uses 8502)

## One-shot bring-up

```powershell
make dev_staging_up
```

This runs: infra bootstrap → k3d cluster ensure → apply manifests.

## Step by step

```powershell
make dev_infra          # Postgres, MinIO, Redis on localhost ports
make k8s_ensure         # create/start k3d cluster chess-teacher
make k8s_up             # apply namespace, secrets, streamlit, cronjobs
make streamlit_k8s      # port-forward http://localhost:8501
```

## What gets deployed

| Resource | Schedule / role |
|----------|-----------------|
| `deployment/streamlit` | UI (image `:develop`) |
| `cronjob/ingestion-dispatcher` | Every 30 min → spawns pipeline Jobs |
| `cronjob/nightly-maintenance` | Daily 03:00 Europe/Amsterdam |

Pipeline Jobs are created at runtime by the dispatcher (same as prod).

## k3d overrides (not in Doppler)

Applied by `apply-k3d-local.ps1` before `apply.ps1`:

| Variable | Value |
|----------|--------|
| `POSTGRES_HOST` | `host.k3d.internal` |
| `S3_ENDPOINT_URL` | `http://host.k3d.internal:9000` |
| `REDIS_URL` | `redis://host.k3d.internal:6379/0` |
| `LOG_BUFFER_DIR` | `/tmp/chess-teacher-logs` |
| `STREAMLIT_REDIRECT_URI` | `http://localhost:8501/oauth2callback` |

## Laptop closed

CronJobs only run while Docker + k3d are up. Data persists under `storage/postgres` and `storage/minio`.

## Refresh after develop deploy

CD rebuilds `:develop` on push. Re-apply or restart:

```powershell
make k8s_up
# or
kubectl rollout restart deployment/streamlit -n chess-teacher
```

## Troubleshooting

**API unreachable:** `make k8s_ensure` or delete and recreate:

```powershell
k3d cluster delete chess-teacher
make k8s_ensure
make k8s_up
```

**Postgres preflight fails:** run `make dev_infra` first.

**Manual dispatcher tick:**

```powershell
doppler run --project chess-teacher --config dev_local -- python scripts/entrypoints/dispatcher.py
```

(Requires kubeconfig pointed at k3d and in-cluster RBAC from `make k8s_up`.)
