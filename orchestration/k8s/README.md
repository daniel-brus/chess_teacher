# Local k3d staging (prod-like)

Mimics production on your laptop: **Compose** for Postgres/MinIO/Redis, **k3d** for Streamlit + CronJobs + pipeline Jobs. Uses the **`develop` Docker image** (`{DOCKERHUB}/chess_teacher:develop`).

No separate Doppler config — **`dev_local` only**. Host routing overrides live in `apply-k3d-local.ps1`.

## Prerequisites

- Docker Desktop running
- `doppler login` + `dev_local` secrets populated (see repo `.env.example`)
- `k3d`, `kubectl` on PATH
- Google OAuth: add redirect URI `http://localhost:8501/oauth2callback` (k8s port-forward; venv uses 8502)

## Bring-up

```powershell
make dev_k3d_up         # infra (wait healthy) + k3d + apply Streamlit/CronJobs
make streamlit_k3d      # port-forward only → http://localhost:8501
```

`streamlit_k3d` does **not** start Streamlit — the pod is already deployed by `dev_k3d_up`. It only tunnels cluster port 8501 to localhost.

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
make dev_k3d_up
# or
kubectl rollout restart deployment/streamlit -n chess-teacher
```

## Troubleshooting

**API unreachable / stuck on "Waiting for Kubernetes API":**
Windows kubeconfig often gets `host.docker.internal`, which kubectl cannot reach.
`make k8s_ensure` / `dev_k3d_up` now bind the API to `127.0.0.1:6550` and rewrite the kubeconfig.
If an old cluster is stuck:

```powershell
k3d cluster delete chess-teacher
make k8s_ensure
make dev_k3d_up
```

**Postgres preflight fails:** run `make dev_bootstrap` first.

**Manual dispatcher tick:**

```powershell
doppler run --project chess-teacher --config dev_local -- python scripts/entrypoints/dispatcher.py
```

(Requires kubeconfig pointed at k3d and in-cluster RBAC from `make dev_k3d_up`.)
