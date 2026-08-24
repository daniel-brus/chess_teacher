#!/usr/bin/env bash
# Launch a whitelisted one-off script Job on the cluster (prod VPS friendly).
#
# Deployed by CD to /opt/chess_teacher/k8s/run-script-job.sh (see .github/workflows/cd.yml).
# Requires only kubectl on the host — no git checkout, no Python, no IMAGE export.
#
# Usage:
#   /opt/chess_teacher/k8s/run-script-job.sh baseline_training
#   /opt/chess_teacher/k8s/run-script-job.sh baseline_training --dry-run
#   /opt/chess_teacher/k8s/run-script-job.sh baseline_promotion -- --wait
#   /opt/chess_teacher/k8s/run-script-job.sh maintenance -- --follow
#
# After `--`, flags are either wrapper controls (--wait / --follow) or passed through
# to the Python script. Wrapper controls are stripped before the Job args list.
set -euo pipefail

K8S_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${K8S_DIR}/job/script.yaml"
NAMESPACE="${NAMESPACE:-chess-teacher}"

# Keep in sync with scripts/run_script_job.py ALLOWED_SCRIPTS.
ALLOWED_SCRIPTS=(
  baseline_training.py
  baseline_promotion.py
  maintenance.py
)

usage() {
  cat <<EOF
Usage: $(basename "$0") <script> [--dry-run] [--] [script-args...] [--wait] [--follow]

  script     Whitelisted entrypoint (with or without .py), e.g. baseline_training
  --dry-run  Print rendered Job YAML and exit
  --wait     Poll until the Job completes or fails
  --follow   Stream Job logs after create (implies waiting for the pod)

Allowed scripts:
$(printf '  - %s\n' "${ALLOWED_SCRIPTS[@]}")

Image is taken from deploy/streamlit (fallback: configmap PIPELINE_JOB_IMAGE).
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

normalize_script() {
  local name="$1"
  name="${name##*/}"
  [[ -n "$name" ]] || die "script name must not be empty"
  [[ "$name" == *.py ]] || name="${name}.py"
  printf '%s' "$name"
}

is_allowed() {
  local candidate="$1"
  local allowed
  for allowed in "${ALLOWED_SCRIPTS[@]}"; do
    if [[ "$candidate" == "$allowed" ]]; then
      return 0
    fi
  done
  return 1
}

# Escape a string for inclusion in a YAML double-quoted scalar.
yaml_escape() {
  local s="$1"
  s="${s//\\/\\\\}"
  s="${s//\"/\\\"}"
  s="${s//$'\n'/\\n}"
  s="${s//$'\r'/\\r}"
  printf '%s' "$s"
}

render_args_yaml() {
  if [[ $# -eq 0 ]]; then
    printf '[]'
    return
  fi
  local out="["
  local first=1
  local arg escaped
  for arg in "$@"; do
    [[ -n "$arg" ]] || die "script arguments must not be empty strings"
    if [[ "$arg" == *[!A-Za-z0-9_./:@%=+-]* ]]; then
      # Allow common flags/values; reject shell metacharacters.
      case "$arg" in
        *[\;\&\|\`\$\<\>\(\)\{\}\\]*)
          die "disallowed characters in script argument: $arg"
          ;;
      esac
    fi
    escaped="$(yaml_escape "$arg")"
    if [[ $first -eq 1 ]]; then
      first=0
    else
      out+=", "
    fi
    out+="\"${escaped}\""
  done
  out+="]"
  printf '%s' "$out"
}

job_name_for() {
  local stem="${1%.py}"
  local stamp
  stamp="$(date -u +%Y%m%d%H%M%S)"
  local raw="script-${stem}-${stamp}"
  # DNS-1123 subdomain: lowercase, digits, hyphens.
  raw="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9-]+/-/g; s/^-+//; s/-+$//')"
  printf '%s' "${raw:0:63}" | sed -E 's/-+$//'
}

resolve_image() {
  local image
  if [[ -n "${PIPELINE_JOB_IMAGE:-}" ]]; then
    printf '%s' "$PIPELINE_JOB_IMAGE"
    return
  fi
  image="$(kubectl get deploy streamlit -n "$NAMESPACE" \
    -o jsonpath='{.spec.template.spec.containers[0].image}' 2>/dev/null || true)"
  if [[ -n "$image" ]]; then
    printf '%s' "$image"
    return
  fi
  image="$(kubectl get cm chess-teacher-config -n "$NAMESPACE" \
    -o jsonpath='{.data.PIPELINE_JOB_IMAGE}' 2>/dev/null || true)"
  if [[ -n "$image" ]]; then
    printf '%s' "$image"
    return
  fi
  die "could not resolve image (set PIPELINE_JOB_IMAGE or ensure deploy/streamlit exists)"
}

resolve_pull_policy() {
  if [[ -n "${IMAGE_PULL_POLICY:-}" ]]; then
    printf '%s' "$IMAGE_PULL_POLICY"
    return
  fi
  local policy
  policy="$(kubectl get cm chess-teacher-config -n "$NAMESPACE" \
    -o jsonpath='{.data.IMAGE_PULL_POLICY}' 2>/dev/null || true)"
  printf '%s' "${policy:-Always}"
}

wait_for_job() {
  local job_name="$1"
  local deadline=$((SECONDS + 86400))
  while (( SECONDS < deadline )); do
    local succ fail
    succ="$(kubectl get job "$job_name" -n "$NAMESPACE" \
      -o jsonpath='{.status.succeeded}' 2>/dev/null || true)"
    fail="$(kubectl get job "$job_name" -n "$NAMESPACE" \
      -o jsonpath='{.status.failed}' 2>/dev/null || true)"
    if [[ -n "$succ" && "$succ" != "0" ]]; then
      echo "Job ${job_name} finished with status: Complete"
      return 0
    fi
    if [[ -n "$fail" && "$fail" != "0" ]]; then
      echo "Job ${job_name} finished with status: Failed" >&2
      return 1
    fi
    sleep 10
  done
  die "timed out waiting for job ${job_name}"
}

follow_logs() {
  local job_name="$1"
  echo "Waiting for pod of job/${job_name}..."
  local pod=""
  local i
  for i in $(seq 1 60); do
    pod="$(kubectl get pods -n "$NAMESPACE" \
      -l "job-name=${job_name}" \
      -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
    if [[ -n "$pod" ]]; then
      break
    fi
    sleep 2
  done
  [[ -n "$pod" ]] || die "no pod appeared for job/${job_name}"
  kubectl logs -n "$NAMESPACE" "pod/${pod}" -f
}

# --- main ---

require_cmd kubectl
require_cmd date
require_cmd sed
require_cmd tr

[[ -f "$TEMPLATE" ]] || die "template not found: $TEMPLATE"

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

SCRIPT_INPUT="$1"
shift

DRY_RUN=0
WAIT=0
FOLLOW=0
SCRIPT_ARGS=()

# Parse: optional --dry-run before `--`; after `--`, collect args and peel --wait/--follow.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --)
      shift
      break
      ;;
    --wait|--follow)
      die "$1 must come after -- (e.g. $(basename "$0") SCRIPT -- --wait)"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1 (script args go after --)"
      ;;
  esac
done

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wait)
      WAIT=1
      shift
      ;;
    --follow)
      FOLLOW=1
      shift
      ;;
    *)
      SCRIPT_ARGS+=("$1")
      shift
      ;;
  esac
done

SCRIPT_BASENAME="$(normalize_script "$SCRIPT_INPUT")"
is_allowed "$SCRIPT_BASENAME" || die "script ${SCRIPT_BASENAME!r} is not whitelisted. Allowed: ${ALLOWED_SCRIPTS[*]}"

JOB_NAME="$(job_name_for "$SCRIPT_BASENAME")"
IMAGE="$(resolve_image)"
PULL_POLICY="$(resolve_pull_policy)"
ARGS_YAML="$(render_args_yaml "${SCRIPT_ARGS[@]+"${SCRIPT_ARGS[@]}"}")"

RENDERED="$(
  sed \
    -e "s|REPLACE_JOB_NAME|${JOB_NAME}|g" \
    -e "s|REPLACE_SCRIPT_JOB_IMAGE|${IMAGE}|g" \
    -e "s|REPLACE_IMAGE_PULL_POLICY|${PULL_POLICY}|g" \
    -e "s|REPLACE_SCRIPT_BASENAME|${SCRIPT_BASENAME}|g" \
    -e "s|REPLACE_SCRIPT_ARGS|${ARGS_YAML}|g" \
    "$TEMPLATE"
)"

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf '%s\n' "$RENDERED"
  exit 0
fi

printf '%s\n' "$RENDERED" | kubectl apply -f - -n "$NAMESPACE"

echo "Created Job ${JOB_NAME} in namespace ${NAMESPACE}."
echo "  image: ${IMAGE}"
echo "  kubectl logs -n ${NAMESPACE} job/${JOB_NAME} -f"
echo "  kubectl delete job ${JOB_NAME} -n ${NAMESPACE}"

if [[ "$FOLLOW" -eq 1 ]]; then
  follow_logs "$JOB_NAME"
  wait_for_job "$JOB_NAME"
elif [[ "$WAIT" -eq 1 ]]; then
  wait_for_job "$JOB_NAME"
fi
