$ErrorActionPreference = "Stop"

$k8sDir = $PSScriptRoot

function Get-RequiredEnv {
    param([string]$Key)
    $value = [Environment]::GetEnvironmentVariable($Key)
    if ([string]::IsNullOrWhiteSpace($value)) {
        throw "Missing required environment variable: $Key (expected from Doppler config dev_local)"
    }
    return $value.Trim()
}

function Get-OptionalEnv {
    param([string]$Key)
    $value = [Environment]::GetEnvironmentVariable($Key)
    if ([string]::IsNullOrWhiteSpace($value)) {
        return $null
    }
    return $value.Trim()
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

function New-StreamlitSecretsToml {
    $redirectUri = Get-RequiredEnv "STREAMLIT_REDIRECT_URI"
    $cookieSecret = Get-RequiredEnv "STREAMLIT_COOKIE_SECRET"
    $clientId = Get-RequiredEnv "STREAMLIT_GOOGLE_CLIENT_ID"
    $clientSecret = Get-RequiredEnv "STREAMLIT_GOOGLE_CLIENT_SECRET"
    return @"
[auth]
redirect_uri = "$redirectUri"
cookie_secret = "$cookieSecret"

[auth.google]
client_id = "$clientId"
client_secret = "$clientSecret"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
"@
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

$dockerhub = Get-RequiredEnv "DOCKERHUB_USERNAME"
$image = "$dockerhub/chess_teacher:develop"

function Render-K8sManifest {
    param([string]$Content, [string]$Image, [string]$PullPolicy, [string]$Environment)
    return $Content.Replace("IMAGE_PLACEHOLDER", $Image).Replace("IMAGE_PULL_POLICY_PLACEHOLDER", $PullPolicy).Replace("ENVIRONMENT_PLACEHOLDER", $Environment)
}

Write-Host "Using image: $image (pull: Always)" -ForegroundColor Cyan
Write-Host "Secrets source: Doppler config dev_local (k3d host overrides from apply-k3d-local.ps1)" -ForegroundColor Cyan

$postgresHost = Get-RequiredEnv "POSTGRES_HOST"
$postgresPort = [int](Get-RequiredEnv "POSTGRES_PORT")
if ($postgresHost -eq "host.k3d.internal") {
    Write-Host "Preflight: checking local Compose Postgres at ${postgresHost}:${postgresPort}..." -ForegroundColor Cyan
    $reachable = Test-NetConnection -ComputerName $postgresHost -Port $postgresPort -WarningAction SilentlyContinue
    if (-not $reachable.TcpTestSucceeded) {
        throw @"
Cannot reach Postgres at ${postgresHost}:${postgresPort}.
Start local infra first: make dev_infra
Then retry: make k8s_up
"@
    }
    Write-Host "Local infra reachable." -ForegroundColor Green
}

Invoke-Kubectl @("apply", "-f", (Join-Path $k8sDir "namespace.yaml"))

$configPath = Join-Path $k8sDir "configmap.yaml"
$config = Render-K8sManifest (Get-Content $configPath -Raw) $image "Always" "DEV"
Invoke-Kubectl @("apply", "-f", "-") -InputObject $config

Invoke-Kubectl @("apply", "-f", (Join-Path $k8sDir "rbac.yaml"))

$postgresSslMode = Get-OptionalEnv POSTGRES_SSLMODE
if ($null -eq $postgresSslMode) {
    $postgresSslMode = ""
}

$secretArgs = @(
    "create", "secret", "generic", "chess-teacher-env",
    "--from-literal=POSTGRES_HOST=$(Get-RequiredEnv POSTGRES_HOST)",
    "--from-literal=POSTGRES_PORT=$(Get-RequiredEnv POSTGRES_PORT)",
    "--from-literal=POSTGRES_DB=$(Get-RequiredEnv POSTGRES_DB)",
    "--from-literal=POSTGRES_USER=$(Get-RequiredEnv POSTGRES_USER)",
    "--from-literal=POSTGRES_PASSWORD=$(Get-RequiredEnv POSTGRES_PASSWORD)",
    "--from-literal=POSTGRES_SSLMODE=$postgresSslMode",
    "--from-literal=STORAGE_ROOT=$(Get-RequiredEnv STORAGE_ROOT)",
    "--from-literal=S3_BUCKET=$(Get-RequiredEnv S3_BUCKET)",
    "--from-literal=S3_ENDPOINT_URL=$(Get-RequiredEnv S3_ENDPOINT_URL)",
    "--from-literal=S3_ACCESS_KEY_ID=$(Get-RequiredEnv S3_ACCESS_KEY_ID)",
    "--from-literal=S3_SECRET_ACCESS_KEY=$(Get-RequiredEnv S3_SECRET_ACCESS_KEY)",
    "--from-literal=LOG_BUFFER_DIR=$(Get-RequiredEnv LOG_BUFFER_DIR)",
    "--from-literal=REDIS_URL=$(Get-RequiredEnv REDIS_URL)",
    "-n", "chess-teacher",
    "--dry-run=client", "-o", "yaml"
)
foreach ($optionalKey in @("LOG_SHIP_ENABLED", "STOCKFISH_WORKERS", "STOCKFISH_THREADS_PER_ENGINE", "STOCKFISH_HASH_MB")) {
    $optionalValue = Get-OptionalEnv $optionalKey
    if ($null -ne $optionalValue) {
        $secretArgs += "--from-literal=${optionalKey}=$optionalValue"
    }
}
$secretYaml = & kubectl @secretArgs
if ($LASTEXITCODE -ne 0) {
    throw "kubectl failed: kubectl create secret generic chess-teacher-env"
}
Invoke-Kubectl @("apply", "-f", "-") -InputObject $secretYaml

foreach ($cronFile in @("nightly-maintenance.yaml", "ingestion-dispatcher.yaml")) {
    $path = Join-Path $k8sDir "cronjob\$cronFile"
    $manifest = Render-K8sManifest (Get-Content $path -Raw) $image "Always" "DEV"
    Invoke-Kubectl @("apply", "-f", "-") -InputObject $manifest
}

$streamlitSecretsFile = Join-Path ([System.IO.Path]::GetTempPath()) "chess-teacher-streamlit-secrets.toml"
try {
    New-StreamlitSecretsToml | Set-Content -Path $streamlitSecretsFile -Encoding UTF8
    $streamlitSecretYaml = & kubectl create secret generic chess-teacher-streamlit-secrets `
        --from-file=secrets.toml=$streamlitSecretsFile `
        -n chess-teacher `
        --dry-run=client -o yaml
    if ($LASTEXITCODE -ne 0) {
        throw "kubectl failed: kubectl create secret generic chess-teacher-streamlit-secrets"
    }
    Invoke-Kubectl @("apply", "-f", "-") -InputObject $streamlitSecretYaml
} finally {
    if (Test-Path $streamlitSecretsFile) {
        Remove-Item $streamlitSecretsFile -Force
    }
}

$streamlitPath = Join-Path $k8sDir "deployment\streamlit.yaml"
$streamlitManifest = Render-K8sManifest (Get-Content $streamlitPath -Raw) $image "Always" "DEV"
Invoke-Kubectl @("apply", "-f", "-") -InputObject $streamlitManifest

Write-Host "Restarting Streamlit rollout..." -ForegroundColor Cyan
Invoke-Kubectl @("rollout", "restart", "deployment/streamlit", "-n", "chess-teacher")
Invoke-Kubectl @("rollout", "status", "deployment/streamlit", "-n", "chess-teacher", "--timeout=300s")

Write-Host ""
Write-Host "Kubernetes orchestration applied." -ForegroundColor Green
Write-Host "App secrets from Doppler dev_local -> chess-teacher-env" -ForegroundColor Green
Write-Host ""
Write-Host "Streamlit (dev image): kubectl port-forward --address 0.0.0.0 -n chess-teacher svc/streamlit 8501:8501" -ForegroundColor Cyan
Write-Host "Then open http://localhost:8501" -ForegroundColor Cyan
