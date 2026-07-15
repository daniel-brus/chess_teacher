# Start local infra and wait for Postgres, MinIO, and Redis to become healthy.
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $Root

Write-Host "Starting local infra (make dev_infra)..." -ForegroundColor Cyan
& make dev_infra
if ($LASTEXITCODE -ne 0) {
    throw "make dev_infra failed"
}

$composeInfra = Join-Path $Root "docker-compose.infra.yml"
$services = @("postgres", "redis")
$maxAttempts = 30

foreach ($service in $services) {
    Write-Host "Waiting for $service to be healthy..." -ForegroundColor DarkGray
    $healthy = $false
    for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
        $raw = & docker --context desktop-linux compose -f $composeInfra ps --format json $service 2>$null
        if (-not $raw) {
            Start-Sleep -Seconds 2
            continue
        }
        $status = $raw | ConvertFrom-Json
        $row = if ($status -is [array]) { $status[0] } else { $status }
        if ($row -and $row.Health -eq "healthy") {
            $healthy = $true
            break
        }
        Start-Sleep -Seconds 2
    }
    if (-not $healthy) {
        throw "Timed out waiting for $service to become healthy."
    }
}

Write-Host "Waiting for minio-init to complete..." -ForegroundColor DarkGray
$initDone = $false
for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
    $initStatus = & docker --context desktop-linux compose -f $composeInfra ps --format json minio-init 2>$null
    if ($initStatus) {
        $row = $initStatus | ConvertFrom-Json
        $state = if ($row -is [array]) { $row[0].State } else { $row.State }
        if ($state -eq "exited") {
            $initDone = $true
            break
        }
    }
    Start-Sleep -Seconds 2
}
if (-not $initDone) {
    throw "Timed out waiting for minio-init to complete."
}

Write-Host "Local infra is ready." -ForegroundColor Green
