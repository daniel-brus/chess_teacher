#!/usr/bin/env bash
# Run the Chess Teacher Streamlit app against the local dev backends.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

set -a
# shellcheck disable=SC1091
. "$REPO_ROOT/.env"
set +a

exec "$REPO_ROOT/.venv/bin/streamlit" run streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port "${APP_PORT:-8502}" \
  --server.headless true
