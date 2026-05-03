# NOTE: This task runs at LOCAL machine time (CET/CEST), not UTC.
# Scheduled via Windows Task Scheduler.

# Access compose files via absolute path in project root (UNNECCESSARY, but for clarity)
$project = Split-Path -Parent $PSScriptRoot

$compose = "-f deploy/docker-compose.yml -f deploy/docker-compose.dev.yml"

$composeCmd = @"
cd `"$project`";

Write-Host "Running nightly job via Docker Compose..." -ForegroundColor Cyan

docker compose $compose up --abort-on-container-exit --exit-code-from nightly-job

Write-Host "Nightly job finished." -ForegroundColor Green
"@

$action = New-ScheduledTaskAction `
  -Execute "powershell.exe" `
  -Argument "-NoProfile -ExecutionPolicy Bypass -Command $composeCmd"

$trigger = New-ScheduledTaskTrigger -Daily -At 3:00AM

$settings = New-ScheduledTaskSettingsSet `
  -WakeToRun `
  -StartWhenAvailable

Register-ScheduledTask `
  -TaskName "ChessTeacherNightlyDev" `
  -Action $action `
  -Trigger $trigger `
  -Settings $settings `
  -Force
