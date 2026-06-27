$ErrorActionPreference = "Stop"

$project = (Resolve-Path "$PSScriptRoot\..\..").Path
$envFile = Join-Path $project ".env"
$k8sDir = $PSScriptRoot

if (-not (Test-Path $envFile)) {
    throw "Missing .env at $envFile"
}

function Read-DotEnvValue {
    param([string]$Key)
    $pattern = "^\s*$([regex]::Escape($Key))\s*=\s*(.+?)\s*$"
    foreach ($line in Get-Content $envFile) {
        if ($line -match $pattern) {
            return $Matches[1].Trim().Trim('"').Trim("'")
        }
    }
    return $null
}

function Invoke-Kubectl {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Args,
        $InputObject
    )
    if ($null -ne $InputObject) {
        $stdin = if ($InputObject -is [string]) { $InputObject } else { $InputObject -join "`n" }
        $stdin | & kubectl @Args
    } else {
        & kubectl @Args
    }
    if ($LASTEXITCODE -ne 0) {
        throw "kubectl failed: kubectl $($Args -join ' ')"
    }
}

Write-Host "Checking Kubernetes cluster connectivity..." -ForegroundColor Cyan
try {
    Invoke-Kubectl @("cluster-info")
} catch {
    throw @"
Kubernetes API is not reachable. Recover the cluster first:
  make k8s_ensure
If that prints manual recreate steps, run them, then:
  make k8s_up
"@
}

$dockerhub = Read-DotEnvValue "DOCKERHUB_USERNAME"
if (-not $dockerhub) {
    throw "DOCKERHUB_USERNAME is missing from .env"
}

$image = "$dockerhub/chess_teacher:develop"
$postgresPort = Read-DotEnvValue "POSTGRES_PORT"
if (-not $postgresPort) {
    $postgresPort = "5432"
}
$appPort = Read-DotEnvValue "APP_PORT"
if (-not $appPort) {
    $appPort = "8501"
}

function Render-K8sManifest {
    param([string]$Content, [string]$Image, [string]$PullPolicy)
    return $Content.Replace("IMAGE_PLACEHOLDER", $Image).Replace("IMAGE_PULL_POLICY_PLACEHOLDER", $PullPolicy)
}

Write-Host "Using image: $image (pull: Always)" -ForegroundColor Cyan

Invoke-Kubectl @("apply", "-f", (Join-Path $k8sDir "namespace.yaml"))

$configPath = Join-Path $k8sDir "configmap.yaml"
$config = Render-K8sManifest (Get-Content $configPath -Raw) $image "Always"
Invoke-Kubectl @("apply", "-f", "-") -InputObject $config

Invoke-Kubectl @("apply", "-f", (Join-Path $k8sDir "rbac.yaml"))

$secretYaml = & kubectl create secret generic chess-teacher-env `
    --from-env-file=$envFile `
    -n chess-teacher `
    --dry-run=client -o yaml
if ($LASTEXITCODE -ne 0) {
    throw "kubectl failed: kubectl create secret generic chess-teacher-env"
}
Invoke-Kubectl @("apply", "-f", "-") -InputObject $secretYaml

foreach ($cronFile in @("nightly-maintenance.yaml", "ingestion-dispatcher.yaml")) {
    $path = Join-Path $k8sDir "cronjob\$cronFile"
    $manifest = Render-K8sManifest (Get-Content $path -Raw) $image "Always"
    Invoke-Kubectl @("apply", "-f", "-") -InputObject $manifest
}

$streamlitSecretsFile = Join-Path $project ".streamlit\secrets.toml"
if (-not (Test-Path $streamlitSecretsFile)) {
    throw "Missing Streamlit secrets at $streamlitSecretsFile (required for OAuth login)."
}
$streamlitSecretYaml = & kubectl create secret generic chess-teacher-streamlit-secrets `
    --from-file=secrets.toml=$streamlitSecretsFile `
    -n chess-teacher `
    --dry-run=client -o yaml
if ($LASTEXITCODE -ne 0) {
    throw "kubectl failed: kubectl create secret generic chess-teacher-streamlit-secrets"
}
Invoke-Kubectl @("apply", "-f", "-") -InputObject $streamlitSecretYaml

$streamlitPath = Join-Path $k8sDir "deployment\streamlit.yaml"
$streamlitManifest = Render-K8sManifest (Get-Content $streamlitPath -Raw) $image "Always"
Invoke-Kubectl @("apply", "-f", "-") -InputObject $streamlitManifest

Write-Host ""
Write-Host "Kubernetes orchestration applied." -ForegroundColor Green
Write-Host ""
Write-Host "Ensure Postgres is reachable from the k3d cluster at the POSTGRES_HOST in configmap.yaml." -ForegroundColor Yellow
Write-Host "Compose Postgres must publish port $postgresPort to the host." -ForegroundColor Yellow
Write-Host ""
Write-Host "Streamlit (dev image): kubectl port-forward -n chess-teacher svc/streamlit ${appPort}:8501" -ForegroundColor Cyan
Write-Host "Then open http://localhost:${appPort}" -ForegroundColor Cyan
