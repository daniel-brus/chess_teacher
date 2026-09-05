#!/usr/bin/env bash
# Idempotently start the local dev backends (Postgres, Redis, MinIO) and ensure
# the app role/database/bucket exist. Safe to run repeatedly (install + every boot).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PG_USER="${POSTGRES_USER:-chess_teacher}"
PG_PASSWORD="${POSTGRES_PASSWORD:-chess_teacher}"
PG_DB="${POSTGRES_DB:-chess_teacher}"
S3_BUCKET="${S3_BUCKET:-chess-teacher}"
S3_ACCESS_KEY_ID="${S3_ACCESS_KEY_ID:-minioadmin}"
S3_SECRET_ACCESS_KEY="${S3_SECRET_ACCESS_KEY:-minioadmin}"
MINIO_DATA_DIR="${MINIO_DATA_DIR:-$REPO_ROOT/storage/minio}"

echo "[start-services] Starting PostgreSQL..."
PG_VER="$(pg_lsclusters -h 2>/dev/null | awk 'NR==1{print $1}')"
PG_CLUSTER="$(pg_lsclusters -h 2>/dev/null | awk 'NR==1{print $2}')"
if [ -n "${PG_VER:-}" ]; then
  if ! pg_lsclusters -h | awk 'NR==1{print $4}' | grep -q online; then
    sudo pg_ctlcluster "$PG_VER" "$PG_CLUSTER" start || true
  fi
fi

# Wait for Postgres to accept connections.
for _ in $(seq 1 30); do
  if sudo -u postgres pg_isready -q; then break; fi
  sleep 1
done

echo "[start-services] Ensuring role and database exist..."
sudo -u postgres psql -v ON_ERROR_STOP=1 <<SQL
DO \$\$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='${PG_USER}') THEN
    CREATE ROLE ${PG_USER} LOGIN PASSWORD '${PG_PASSWORD}';
  END IF;
END \$\$;
SQL
if ! sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='${PG_DB}'" | grep -q 1; then
  sudo -u postgres createdb -O "${PG_USER}" "${PG_DB}"
fi
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE ${PG_DB} TO ${PG_USER};" >/dev/null

echo "[start-services] Starting Redis..."
if ! redis-cli ping >/dev/null 2>&1; then
  redis-server --daemonize yes --dir /tmp
fi

echo "[start-services] Starting MinIO..."
mkdir -p "$MINIO_DATA_DIR"
if ! curl -fsS -m 3 http://localhost:9000/minio/health/live >/dev/null 2>&1; then
  MINIO_ROOT_USER="$S3_ACCESS_KEY_ID" MINIO_ROOT_PASSWORD="$S3_SECRET_ACCESS_KEY" \
    nohup /usr/local/bin/minio server "$MINIO_DATA_DIR" \
      --address :9000 --console-address :9001 > /tmp/minio.log 2>&1 &
fi
for _ in $(seq 1 30); do
  if curl -fsS -m 3 http://localhost:9000/minio/health/live >/dev/null 2>&1; then break; fi
  sleep 1
done

echo "[start-services] Ensuring MinIO bucket exists..."
/usr/local/bin/mc alias set local http://localhost:9000 "$S3_ACCESS_KEY_ID" "$S3_SECRET_ACCESS_KEY" >/dev/null 2>&1 || true
/usr/local/bin/mc mb --ignore-existing "local/${S3_BUCKET}" >/dev/null 2>&1 || true

echo "[start-services] All backends are up (Postgres, Redis, MinIO)."
