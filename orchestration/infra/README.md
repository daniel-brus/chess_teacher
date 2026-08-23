# Host Postgres on Hetzner (outside k3s)

Self-hosted `postgres:17` on the VPS host. The app still connects via Doppler `POSTGRES_*` → `DatabaseClient` — same as Supabase.

Keep **`POSTGRES_USER` / `POSTGRES_DB`** equal to what you want after cutover (recommended: `postgres` / `postgres`). Do **not** copy Supabase pooler usernames like `postgres.<project-ref>`.

`postgres.env` on the VPS has **no `POSTGRES_HOST`**: that file only starts the container. Clients get the host from Doppler (`POSTGRES_HOST=<VPS node IP>`) after cutover.

**Ops login:** SSH as **`root`**.
**Container runtime:** Ubuntu **`docker.io` + `docker-compose-v2`** (not Docker CE — avoids replacing k3s’s containerd).

## Files

| File | Role |
|------|------|
| `docker-compose.postgres.yml` | Postgres service |
| `postgres.env.example` | Template for `/opt/chess_teacher/infra/postgres.env` |
| `install-postgres.sh` | Create data dir + `compose up` + health wait |

## One-time host install (run on the VPS as root)

### 1. Docker (Ubuntu packages — Option A)

```bash
apt-get update
apt-get install -y docker.io docker-compose-v2
systemctl enable --now docker
docker version
docker compose version
```

### 2. Place infra files

From your laptop (PowerShell), from **this** repo root
(`chess_teacher-codebase-structure`):

```powershell
ssh root@YOUR_VPS_HOST "mkdir -p /opt/chess_teacher/infra"

scp orchestration\infra\docker-compose.postgres.yml root@YOUR_VPS_HOST:/opt/chess_teacher/infra/
scp orchestration\infra\install-postgres.sh root@YOUR_VPS_HOST:/opt/chess_teacher/infra/
scp orchestration\infra\postgres.env.example root@YOUR_VPS_HOST:/opt/chess_teacher/infra/
```

If `./install-postgres.sh` fails with `bash\r`, fix CRLF on the VPS:

```bash
sed -i 's/\r$//' install-postgres.sh postgres.env
```

On the VPS (as root):

```bash
cd /opt/chess_teacher/infra
chmod +x install-postgres.sh
cp postgres.env.example postgres.env
nano postgres.env   # strong POSTGRES_PASSWORD; USER/DB = postgres / postgres
./install-postgres.sh
```

### 3. Smoke test (on VPS)

```bash
cd /opt/chess_teacher/infra
docker compose --env-file postgres.env -f docker-compose.postgres.yml ps
docker compose --env-file postgres.env -f docker-compose.postgres.yml exec -T postgres pg_isready
docker compose --env-file postgres.env -f docker-compose.postgres.yml exec -T postgres \
  psql -U postgres -d postgres -c "SELECT version();"
hostname -I | awk '{print $1}'   # save for POSTGRES_HOST at cutover
```

### 4. Firewall

Prefer **no public 5432**. Use SSH tunnel for admin/`pg_restore`.
Hetzner Cloud Firewall: do not allow `0.0.0.0/0` on 5432. Same-host k3s→Postgres traffic usually does not need a cloud-firewall pod-CIDR rule.

## Dump Supabase → restore VPS

On your laptop (Doppler `prod` still pointing at Supabase). Without local `psql`/`pg_dump`, use Docker:

```powershell
# Dump (custom format) — runs pg_dump in a container
docker run --rm -v ${PWD}:/out postgres:17 `
  pg_dump "host=$(doppler secrets get POSTGRES_HOST --project chess-teacher --config prod --plain) port=$(doppler secrets get POSTGRES_PORT --project chess-teacher --config prod --plain) dbname=$(doppler secrets get POSTGRES_DB --project chess-teacher --config prod --plain) user=$(doppler secrets get POSTGRES_USER --project chess-teacher --config prod --plain) password=$(doppler secrets get POSTGRES_PASSWORD --project chess-teacher --config prod --plain) sslmode=$(doppler secrets get POSTGRES_SSLMODE --project chess-teacher --config prod --plain)" `
  --format=custom --no-owner --no-acl -f /out/chess_teacher.dump
```

Restore via SSH tunnel:

```powershell
# Terminal A
ssh -L 5433:127.0.0.1:5432 root@YOUR_VPS_HOST

# Terminal B — password from VPS postgres.env
docker run --rm -v ${PWD}:/out -e PGPASSWORD=YOUR_VPS_PASSWORD postgres:17 `
  pg_restore -h host.docker.internal -p 5433 -U postgres -d postgres `
  --no-owner --no-acl --verbose /out/chess_teacher.dump
```

If restore fails on missing extensions, create them first (same tunnel + Docker `psql`):

```powershell
docker run --rm -e PGPASSWORD=YOUR_VPS_PASSWORD postgres:17 `
  psql -h host.docker.internal -p 5433 -U postgres -d postgres `
  -c 'CREATE EXTENSION IF NOT EXISTS "uuid-ossp"; CREATE EXTENSION IF NOT EXISTS pgcrypto;'
```

## Doppler cutover (after restore looks good)

Update **host / password / user / SSL** as needed — keep DB name if it already matches:

```bash
doppler secrets set POSTGRES_HOST=YOUR_NODE_IP --project chess-teacher --config prod
doppler secrets set POSTGRES_USER=postgres --project chess-teacher --config prod
doppler secrets set POSTGRES_PASSWORD='the-password-in-postgres.env' --project chess-teacher --config prod
doppler secrets set POSTGRES_SSLMODE=prefer --project chess-teacher --config prod
# POSTGRES_DB=postgres if that is already correct; leave PORT if still 5432
```

Re-apply k8s secrets:

```bash
cd /opt/chess_teacher/k8s && doppler run --project chess-teacher --config prod -- ./apply.sh
```

Then restart Streamlit so pods reload the secret.

## App connectivity

No code changes. Same mechanism as today:

`POSTGRES_HOST` / `PORT` / `DB` / `USER` / `PASSWORD` / `SSLMODE` → SQLAlchemy engine → `DatabaseClient`.
