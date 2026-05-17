$project = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $project

# REMARK: no streamlit app (use flag --profile streamlit)
$composeFiles = @(
  "-f", "orchestration/docker/docker-compose.yml",
  "-f", "orchestration/docker/docker-compose.dev.yml"
)

$logDir = Join-Path $project "storage/raw/logs/nightly_job"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$logFile = Join-Path $logDir "nightly_job_$timestamp.log"

Start-Transcript -Path $logFile

try {
    Write-Host "Running nightly job via Docker Compose..." -ForegroundColor Cyan

    docker compose @composeFiles --env-file "$project\.env" up --abort-on-container-exit --exit-code-from nightly-job

    Write-Host "Nightly job finished." -ForegroundColor Green
}
finally {
    Write-Host "Cleaning up old log files (>30 days)..." -ForegroundColor Yellow

    Get-ChildItem -Path $logDir -Filter *.log |
        Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } |
        Remove-Item -Force

    Write-Host "Log cleanup completed." -ForegroundColor Green

    Stop-Transcript
}
