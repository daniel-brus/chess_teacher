# Apply k8s manifests for local staging (k3d + Compose infra on the host).
# Uses Doppler dev_local for secrets, then overrides only what pods cannot reach via localhost.
$ErrorActionPreference = "Stop"

$K3D_OVERRIDES = @{
    POSTGRES_HOST           = "host.k3d.internal"
    S3_ENDPOINT_URL         = "http://host.k3d.internal:9000"
    REDIS_URL               = "redis://host.k3d.internal:6379/0"
    LOG_BUFFER_DIR          = "/tmp/chess-teacher-logs"
    STREAMLIT_REDIRECT_URI  = "http://localhost:8501/oauth2callback"
}

foreach ($key in $K3D_OVERRIDES.Keys) {
    [Environment]::SetEnvironmentVariable($key, $K3D_OVERRIDES[$key], "Process")
}

Write-Host "k3d local overrides (from dev_local + repo defaults):" -ForegroundColor Cyan
foreach ($key in ($K3D_OVERRIDES.Keys | Sort-Object)) {
    Write-Host "  $key=$($K3D_OVERRIDES[$key])" -ForegroundColor DarkGray
}

& (Join-Path $PSScriptRoot "apply.ps1")
