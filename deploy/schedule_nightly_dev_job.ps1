# NOTE: This task runs at LOCAL machine time (CET/CEST), not UTC.
# Scheduled via Windows Task Scheduler.

$project = Split-Path -Parent $PSScriptRoot

$compose = "-f deploy/docker-compose.yml -f deploy/docker-compose.dev.yml"

# Create log directory (if it doesn't exist)
$logDir = Join-Path $project "storage/raw/logs/nightly_job"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

# Create timestamped logfile
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$logFile = Join-Path $logDir "nightly_job_$timestamp.log"

$composeCmd = @"
cd `"$project`";

Write-Host "Running nightly job via Docker Compose..." -ForegroundColor Cyan

docker compose --env-file `"$project\.env`" $compose up --abort-on-container-exit --exit-code-from nightly-job 2>&1 |
Tee-Object -FilePath `"$logFile`" -Append

Write-Host "Nightly job finished." -ForegroundColor Green

Write-Host "Cleaning up old log files (>30 days)..." -ForegroundColor Yellow

Get-ChildItem -Path `"$logDir`" -Filter *.log |
Where-Object { \$_.LastWriteTime -lt (Get-Date).AddDays(-30) } |
Remove-Item -Force

Write-Host "Log cleanup completed." -ForegroundColor Green
"@

$action = New-ScheduledTaskAction `
  -Execute "powershell.exe" `
  -Argument "-NoProfile -ExecutionPolicy Bypass -Command $composeCmd"

$trigger = New-ScheduledTaskTrigger -Daily -At 3:00AM

$settings = New-ScheduledTaskSettingsSet `
  -WakeToRun `
  -StartWhenAvailable `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries

Register-ScheduledTask `
  -TaskName "ChessTeacherNightlyDev" `
  -Action $action `
  -Trigger $trigger `
  -Settings $settings `
  -Force
