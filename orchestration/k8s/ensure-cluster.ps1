# Ensures the local k3d cluster exists and the Kubernetes API is healthy.
# Auto-recovery: merge kubeconfig -> start -> stop/start.
# If still unhealthy, prints manual delete+recreate steps (does not run them).
param(
    [string]$ClusterName = "chess-teacher"
)

$ErrorActionPreference = "Stop"

$dockerContext = "desktop-linux"

function Invoke-External {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [string[]]$Command
    )
    & $Command[0] $Command[1..($Command.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed: $($Command -join ' ')"
    }
}

function Test-DockerReady {
    & docker --context $dockerContext info *> $null
    return $LASTEXITCODE -eq 0
}

function Test-ClusterExists {
    $clusters = & k3d cluster list --no-headers 2>$null
    if ($LASTEXITCODE -ne 0) {
        return $false
    }
    return [bool]($clusters | Where-Object { $_ -match "^\s*$([regex]::Escape($ClusterName))\s" })
}

function Merge-Kubeconfig {
    if (-not (Test-ClusterExists)) {
        return
    }
    Invoke-External "k3d kubeconfig merge" @(
        "k3d", "kubeconfig", "merge", $ClusterName, "-d", "-u", "--overwrite"
    )
}

function Test-ApiHealthy {
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & kubectl --request-timeout=5s get nodes --no-headers 2>$null
        if ($LASTEXITCODE -ne 0) {
            return $false
        }
        return ($output | Where-Object { $_ -match "\sReady\s" }).Count -gt 0
    } finally {
        $ErrorActionPreference = $previousErrorAction
    }
}

function Wait-ApiHealthy {
    param(
        [int]$MaxAttempts = 30,
        [int]$DelaySeconds = 2
    )
    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        Merge-Kubeconfig
        if (Test-ApiHealthy) {
            return $true
        }
        Write-Host "Waiting for Kubernetes API ($attempt/$MaxAttempts)..." -ForegroundColor DarkGray
        Start-Sleep -Seconds $DelaySeconds
    }
    return $false
}

function Get-CreateArgs {
    return @(
        "k3d", "cluster", "create", $ClusterName,
        "--kubeconfig-update-default",
        "--wait"
    )
}

function Format-CreateCommand {
    return (Get-CreateArgs) -join " "
}

function New-Cluster {
    Write-Host "Creating k3d cluster '$ClusterName'..." -ForegroundColor Cyan
    Invoke-External "k3d cluster create" (Get-CreateArgs)
    Merge-Kubeconfig
    if (-not (Wait-ApiHealthy)) {
        throw "Cluster '$ClusterName' was created but the Kubernetes API is still unreachable."
    }
}

function Start-Cluster {
    Write-Host "Starting k3d cluster '$ClusterName'..." -ForegroundColor Cyan
    Invoke-External "k3d cluster start" @("k3d", "cluster", "start", $ClusterName)
}

function Restart-Cluster {
    Write-Host "Restarting k3d cluster '$ClusterName'..." -ForegroundColor Yellow
    Invoke-External "k3d cluster stop" @("k3d", "cluster", "stop", $ClusterName)
    Start-Cluster
}

function Write-RecreateInstructions {
    $createCommand = Format-CreateCommand
    Write-Host ""
    Write-Host "Cluster '$ClusterName' is still unreachable after automatic recovery." -ForegroundColor Red
    Write-Host ""
    Write-Host "Recreate it manually, then rerun:" -ForegroundColor Yellow
    Write-Host "  make k8s_up"
    Write-Host ""
    Write-Host "  k3d cluster delete $ClusterName"
    Write-Host "  $createCommand"
    Write-Host ""
    exit 1
}

Write-Host "Ensuring k3d cluster '$ClusterName' is healthy..." -ForegroundColor Cyan

if (-not (Test-DockerReady)) {
    throw @"
Docker Desktop is not reachable via context '$dockerContext'.
Start Docker Desktop, wait until it is ready, then retry.
"@
}

if (-not (Test-ClusterExists)) {
    New-Cluster
    Write-Host "Cluster '$ClusterName' is ready." -ForegroundColor Green
    exit 0
}

Merge-Kubeconfig
if (Test-ApiHealthy) {
    Write-Host "Cluster '$ClusterName' is already healthy." -ForegroundColor Green
    exit 0
}

Write-Host "Cluster exists but API is unreachable; attempting recovery..." -ForegroundColor Yellow

try {
    Start-Cluster
} catch {
    Write-Host "Start failed: $($_.Exception.Message)" -ForegroundColor DarkYellow
}

if (Wait-ApiHealthy -MaxAttempts 15) {
    Write-Host "Cluster '$ClusterName' recovered after start." -ForegroundColor Green
    exit 0
}

try {
    Restart-Cluster
} catch {
    Write-Host "Restart failed: $($_.Exception.Message)" -ForegroundColor DarkYellow
}

if (Wait-ApiHealthy -MaxAttempts 20) {
    Write-Host "Cluster '$ClusterName' recovered after restart." -ForegroundColor Green
    exit 0
}

Write-RecreateInstructions
