#!/usr/bin/env bash
# Create dirs and start host Postgres on the VPS (outside k3s).
# Intended to run as root on the Hetzner host (Ubuntu docker.io + compose v2).
set -euo pipefail

INFRA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${INFRA_DIR}/postgres.env"
COMPOSE_FILE="${INFRA_DIR}/docker-compose.postgres.yml"
DATA_DIR_DEFAULT="/var/lib/chess_teacher/postgres"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}"
  echo "Copy postgres.env.example → postgres.env and set POSTGRES_PASSWORD (keep USER/DB as in Doppler prod)."
  exit 1
fi

DATA_DIR="$(
  grep -E '^POSTGRES_DATA_DIR=' "${ENV_FILE}" | tail -n1 | cut -d= -f2- | tr -d '\r' || true
)"
DATA_DIR="${DATA_DIR:-${DATA_DIR_DEFAULT}}"

echo "==> Ensuring data directory ${DATA_DIR}"
mkdir -p "${DATA_DIR}"
chown -R 999:999 "${DATA_DIR}"

echo "==> Starting Postgres"
cd "${INFRA_DIR}"
docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" up -d

echo "==> Waiting for healthy"
for _ in $(seq 1 30); do
  if docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" exec -T postgres \
    pg_isready >/dev/null 2>&1; then
    echo "Postgres is ready."
    docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" ps
    exit 0
  fi
  sleep 1
done

echo "Timed out waiting for pg_isready"
docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" logs --tail 50
exit 1
