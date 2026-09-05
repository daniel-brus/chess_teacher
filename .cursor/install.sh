#!/usr/bin/env bash
# Repository bootstrap for the Chess Teacher dev environment.
# Idempotent: creates the venv, installs Python deps, writes local dev config
# (.env + Streamlit secrets), starts the local backends and bootstraps the DB schema.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="$REPO_ROOT/.venv"
PY="$VENV_DIR/bin/python"

echo "[install] Creating virtualenv (Python 3.12)..."
if [ ! -x "$PY" ]; then
  python3.12 -m venv "$VENV_DIR"
fi

echo "[install] Installing Python dependencies..."
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r requirements-dev.txt
"$PY" -m pip install -e .

# Local dev config. Doppler is used on developer laptops / CI / prod; in the
# Cloud Agent we generate a self-contained .env pointing at local backends.
if [ ! -f "$REPO_ROOT/.env" ]; then
  echo "[install] Writing local .env..."
  cat > "$REPO_ROOT/.env" <<'ENV'
APP_PORT=8502
ENVIRONMENT=DEV
HOSTNAME=chess-teacher-cloud-dev

POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=chess_teacher
POSTGRES_USER=chess_teacher
POSTGRES_PASSWORD=chess_teacher
POSTGRES_SSLMODE=

S3_BUCKET=chess-teacher
S3_ENDPOINT_URL=http://localhost:9000
S3_ACCESS_KEY_ID=minioadmin
S3_SECRET_ACCESS_KEY=minioadmin
STORAGE_ROOT=chess-teacher

REDIS_URL=redis://localhost:6379/0
LOG_BUFFER_DIR=./storage/log-buffer
LOG_SHIP_ENABLED=false

STOCKFISH_WORKERS=2
STOCKFISH_PATH=/usr/games/stockfish

MLFLOW_EXPERIMENT_NAME=baseline

STREAMLIT_REDIRECT_URI=http://localhost:8502/oauth2callback
STREAMLIT_COOKIE_SECRET=local-dev-cookie-secret-not-for-prod
STREAMLIT_GOOGLE_CLIENT_ID=local-dev.apps.googleusercontent.com
STREAMLIT_GOOGLE_CLIENT_SECRET=local-dev-client-secret
ENV
fi

# Render Streamlit auth secrets from the .env values (Google OAuth placeholders
# for local dev — real credentials come from Doppler in production).
set -a
# shellcheck disable=SC1091
. "$REPO_ROOT/.env"
set +a
echo "[install] Rendering .streamlit/secrets.toml..."
"$PY" scripts/dev/render_streamlit_secrets.py

# Bring up backends and create the schema so the app is usable immediately.
bash "$REPO_ROOT/.cursor/start-services.sh"

echo "[install] Bootstrapping database schema..."
"$PY" scripts/dev/bootstrap_schema.py

echo "[install] Done."
