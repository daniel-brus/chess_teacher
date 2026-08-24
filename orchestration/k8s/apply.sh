#!/usr/bin/env bash
# Production k8s deploy (Hetzner). Invoked by GitHub Actions after SCP of orchestration/k8s/.
set -euo pipefail

K8S_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="chess-teacher"

IMAGE="${IMAGE:?IMAGE is required (e.g. danielbrus/chess_teacher:latest)}"
IMAGE_PULL_POLICY="${IMAGE_PULL_POLICY:-IfNotPresent}"
ENVIRONMENT="${ENVIRONMENT:-PROD}"

REQUIRED_VARS=(
  POSTGRES_HOST
  POSTGRES_PORT
  POSTGRES_DB
  POSTGRES_USER
  POSTGRES_PASSWORD
  POSTGRES_SSLMODE
  STORAGE_ROOT
  S3_BUCKET
  S3_ENDPOINT_URL
  S3_ACCESS_KEY_ID
  S3_SECRET_ACCESS_KEY
  LOG_BUFFER_DIR
  REDIS_URL
  STREAMLIT_REDIRECT_URI
  STREAMLIT_COOKIE_SECRET
  STREAMLIT_GOOGLE_CLIENT_ID
  STREAMLIT_GOOGLE_CLIENT_SECRET
)

OPTIONAL_ENV_VARS=(
  LOG_SHIP_ENABLED
  STOCKFISH_WORKERS
  STOCKFISH_THREADS_PER_ENGINE
  STOCKFISH_HASH_MB
)

log() {
  echo "==> $*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: required command not found: $1" >&2
    exit 1
  fi
}

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: required environment variable missing: $name" >&2
    exit 1
  fi
}

render_manifest() {
  local content="$1"
  content="${content//IMAGE_PLACEHOLDER/$IMAGE}"
  content="${content//IMAGE_PULL_POLICY_PLACEHOLDER/$IMAGE_PULL_POLICY}"
  content="${content//ENVIRONMENT_PLACEHOLDER/$ENVIRONMENT}"
  printf '%s' "$content"
}

kubectl_apply() {
  if ! kubectl "$@"; then
    echo "ERROR: kubectl $* failed" >&2
    kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' 2>/dev/null | tail -20 || true
    exit 1
  fi
}

on_failure() {
  echo "ERROR: deploy failed — recent chess-teacher events:" >&2
  kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' 2>/dev/null | tail -30 || true
  kubectl get pods -n "$NAMESPACE" -o wide 2>/dev/null || true
}
trap on_failure ERR

require_cmd kubectl

log "Pre-flight checks"
for var in "${REQUIRED_VARS[@]}"; do
  require_var "$var"
done

log "Cluster connectivity"
kubectl_apply cluster-info

if ! kubectl get crd clusterissuers.cert-manager.io >/dev/null 2>&1; then
  echo "ERROR: cert-manager CRDs not found. Install once on the VPS:" >&2
  echo "  kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.17.1/cert-manager.yaml" >&2
  exit 1
fi

log "Deploy settings"
echo "  IMAGE=$IMAGE"
echo "  IMAGE_PULL_POLICY=$IMAGE_PULL_POLICY"
echo "  ENVIRONMENT=$ENVIRONMENT"
echo "  NAMESPACE=$NAMESPACE"

log "Apply namespace"
kubectl_apply apply -f "$K8S_DIR/namespace.yaml"

log "Apply ConfigMap"
config_content="$(render_manifest "$(cat "$K8S_DIR/configmap.yaml")")"
printf '%s\n' "$config_content" | kubectl_apply apply -f -

log "Apply RBAC"
kubectl_apply apply -f "$K8S_DIR/rbac.yaml"

log "Apply ClusterIssuer"
if [[ -z "${CERT_MANAGER_ACME_EMAIL:-}" ]]; then
  echo "ERROR: CERT_MANAGER_ACME_EMAIL is required for Let's Encrypt TLS" >&2
  exit 1
fi
issuer_content="$(cat "$K8S_DIR/cert-manager/cluster-issuer.yaml")"
issuer_content="${issuer_content//CERT_MANAGER_ACME_EMAIL_PLACEHOLDER/$CERT_MANAGER_ACME_EMAIL}"
printf '%s\n' "$issuer_content" | kubectl_apply apply -f -

log "Create/update secret chess-teacher-env"
secret_args=(--from-literal="POSTGRES_HOST=${POSTGRES_HOST}")
secret_args+=(--from-literal="POSTGRES_PORT=${POSTGRES_PORT}")
secret_args+=(--from-literal="POSTGRES_DB=${POSTGRES_DB}")
secret_args+=(--from-literal="POSTGRES_USER=${POSTGRES_USER}")
secret_args+=(--from-literal="POSTGRES_PASSWORD=${POSTGRES_PASSWORD}")
secret_args+=(--from-literal="POSTGRES_SSLMODE=${POSTGRES_SSLMODE}")
secret_args+=(--from-literal="STORAGE_ROOT=${STORAGE_ROOT}")
secret_args+=(--from-literal="S3_BUCKET=${S3_BUCKET}")
secret_args+=(--from-literal="S3_ENDPOINT_URL=${S3_ENDPOINT_URL}")
secret_args+=(--from-literal="S3_ACCESS_KEY_ID=${S3_ACCESS_KEY_ID}")
secret_args+=(--from-literal="S3_SECRET_ACCESS_KEY=${S3_SECRET_ACCESS_KEY}")
secret_args+=(--from-literal="LOG_BUFFER_DIR=${LOG_BUFFER_DIR}")
secret_args+=(--from-literal="REDIS_URL=${REDIS_URL}")
for var in "${OPTIONAL_ENV_VARS[@]}"; do
  if [[ -n "${!var:-}" ]]; then
    secret_args+=(--from-literal="${var}=${!var}")
  fi
done
kubectl create secret generic chess-teacher-env \
  "${secret_args[@]}" \
  -n "$NAMESPACE" \
  --dry-run=client -o yaml | kubectl_apply apply -f -

log "Create/update secret chess-teacher-streamlit-secrets"
streamlit_secrets_file="$(mktemp)"
cleanup() {
  rm -f "$streamlit_secrets_file"
}
trap cleanup EXIT
trap on_failure ERR
cat >"$streamlit_secrets_file" <<EOF
[auth]
redirect_uri = "${STREAMLIT_REDIRECT_URI}"
cookie_secret = "${STREAMLIT_COOKIE_SECRET}"

[auth.google]
client_id = "${STREAMLIT_GOOGLE_CLIENT_ID}"
client_secret = "${STREAMLIT_GOOGLE_CLIENT_SECRET}"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
EOF
kubectl create secret generic chess-teacher-streamlit-secrets \
  --from-file=secrets.toml="$streamlit_secrets_file" \
  -n "$NAMESPACE" \
  --dry-run=client -o yaml | kubectl_apply apply -f -

for cron_file in nightly-maintenance.yaml ingestion-dispatcher.yaml; do
  log "Apply CronJob $cron_file"
  cron_content="$(render_manifest "$(cat "$K8S_DIR/cronjob/$cron_file")")"
  printf '%s\n' "$cron_content" | kubectl_apply apply -f -
done

if [[ -f "$K8S_DIR/run-script-job.sh" ]]; then
  chmod +x "$K8S_DIR/run-script-job.sh" || true
fi

log "Apply Streamlit Deployment and Service"
streamlit_content="$(render_manifest "$(cat "$K8S_DIR/deployment/streamlit.yaml")")"
printf '%s\n' "$streamlit_content" | kubectl_apply apply -f -

log "Apply Streamlit Ingress (production TLS)"
kubectl_apply apply -f "$K8S_DIR/deployment/streamlit-ingress.yaml"

log "Restart Streamlit rollout"
kubectl_apply rollout restart "deployment/streamlit" -n "$NAMESPACE"

log "Wait for Streamlit rollout"
kubectl_apply rollout status "deployment/streamlit" -n "$NAMESPACE" --timeout=300s

log "Post-deploy status"
kubectl_apply get pods,cronjobs,ingress -n "$NAMESPACE"

log "Deploy complete"
