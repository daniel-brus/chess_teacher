#!/usr/bin/env bash
# Idempotent local infra for Cloud Agent snapshots: Postgres 17, Redis, MinIO.
# Credentials match .env.example (local dev defaults, not production secrets).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

sudo apt-get update
sudo apt-get install -y \
  python3.12-venv \
  python3.12-dev \
  stockfish \
  ca-certificates \
  curl \
  gnupg \
  redis-server \
  git

if ! command -v psql >/dev/null 2>&1 || ! psql --version | grep -q ' 17\.'; then
  sudo install -d /usr/share/postgresql-common/pgdg
  sudo curl -fsSL -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc \
    https://www.postgresql.org/media/keys/ACCC4CF8.asc
  echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt noble-pgdg main" \
    | sudo tee /etc/apt/sources.list.d/pgdg.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y postgresql-17
fi

if [[ ! -x /usr/local/bin/minio ]]; then
  # Community MinIO binaries are no longer served. Build the last release from source.
  src=/tmp/minio-src
  rm -rf "$src"
  git clone --depth 1 --branch RELEASE.2025-10-15T17-29-55Z \
    https://github.com/minio/minio.git "$src"
  (
    cd "$src"
    GOTOOLCHAIN=go1.25.9 go build -trimpath -o /tmp/minio-bin .
  )
  sudo install -m 755 /tmp/minio-bin /usr/local/bin/minio
  rm -rf "$src" /tmp/minio-bin
fi

sudo install -d -o postgres -g postgres /var/lib/postgresql
sudo install -d -o redis -g redis /var/lib/redis /var/run/redis
sudo install -d -o ubuntu -g ubuntu /var/lib/minio
sudo install -d /var/log/chess-teacher /etc/chess-teacher

sudo tee /etc/profile.d/chess-teacher-infra.sh >/dev/null << 'ENV'
export POSTGRES_HOST=127.0.0.1
export POSTGRES_PORT=5432
export POSTGRES_DB=chess_teacher
export POSTGRES_USER=chess_teacher
export POSTGRES_PASSWORD=change-me
export S3_BUCKET=chess-teacher
export S3_ENDPOINT_URL=http://127.0.0.1:9000
export S3_ACCESS_KEY_ID=minioadmin
export S3_SECRET_ACCESS_KEY=minioadmin
export STORAGE_ROOT=chess-teacher
export REDIS_URL=redis://127.0.0.1:6379/0
export ENVIRONMENT=DEV
export LOG_BUFFER_DIR=/var/lib/chess-teacher/log-buffer
export LOG_SHIP_ENABLED=false
export APP_PORT=8502
export HOSTNAME="${HOSTNAME:-cursor}"
ENV
sudo chmod 644 /etc/profile.d/chess-teacher-infra.sh

# /etc/environment is KEY=VALUE. Refresh the chess-teacher block in place.
sudo touch /etc/environment
sudo python3 - << 'PY'
from pathlib import Path
path = Path("/etc/environment")
keys = {
    "POSTGRES_HOST": "127.0.0.1",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "chess_teacher",
    "POSTGRES_USER": "chess_teacher",
    "POSTGRES_PASSWORD": "change-me",
    "S3_BUCKET": "chess-teacher",
    "S3_ENDPOINT_URL": "http://127.0.0.1:9000",
    "S3_ACCESS_KEY_ID": "minioadmin",
    "S3_SECRET_ACCESS_KEY": "minioadmin",
    "STORAGE_ROOT": "chess-teacher",
    "REDIS_URL": "redis://127.0.0.1:6379/0",
    "ENVIRONMENT": "DEV",
    "LOG_BUFFER_DIR": "/var/lib/chess-teacher/log-buffer",
    "LOG_SHIP_ENABLED": "false",
    "APP_PORT": "8502",
    "HOSTNAME": "cursor",
}
lines = []
if path.exists():
    for line in path.read_text().splitlines():
        key = line.split("=", 1)[0]
        if key not in keys:
            lines.append(line)
for key, value in keys.items():
    lines.append(f"{key}={value}")
path.write_text("\n".join(lines) + "\n")
PY

# Allow password login on localhost TCP. Peer auth on the socket stays for the postgres OS user.
pg_hba="/etc/postgresql/17/main/pg_hba.conf"
if [[ -f "$pg_hba" ]] && ! sudo grep -q "chess-teacher-local" "$pg_hba"; then
  echo "host chess_teacher chess_teacher 127.0.0.1/32 scram-sha-256  # chess-teacher-local" \
    | sudo tee -a "$pg_hba" >/dev/null
fi

echo "chess-teacher infra packages ready"
