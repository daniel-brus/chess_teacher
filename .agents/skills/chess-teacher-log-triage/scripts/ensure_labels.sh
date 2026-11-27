#!/usr/bin/env bash
# Create GitHub labels used by triage/implement automations (idempotent).
set -euo pipefail
labels=(
  "from-logs:#0E8A16:Opened from S3/app log triage"
  "from-vps:#1D76DB:Opened from VPS kubectl log triage"
  "from-ci:#5319E7:Opened from GitHub Actions triage"
  "agent-fix:#D93F0B:Ready for Implement Automation"
  "in-progress:#FBCA04:Implement Automation claimed"
  "pr-open:#C2E0C6:Fix PR opened"
  "wontfix:#ffffff:Do not auto-reopen"
)
for entry in "${labels[@]}"; do
  IFS=':' read -r name color desc <<<"$entry"
  # desc may contain colons — rebuild
  name="${entry%%:*}"
  rest="${entry#*:}"
  color="${rest%%:*}"
  desc="${rest#*:}"
  gh label create "$name" --color "${color#\#}" --description "$desc" 2>/dev/null \
    || gh label edit "$name" --color "${color#\#}" --description "$desc"
done
echo "labels ok"
