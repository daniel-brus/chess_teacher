# Full clone: cloud (Doppler prod) -> local Compose (Doppler dev_local).
# Requires: doppler login, make dev_infra, Docker Desktop, aws CLI on PATH.
# pg_dump/pg_restore run via Docker (no local Postgres client install).
# Client image must be >= cloud server major version (Supabase uses PG 17).
param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$Project = "chess-teacher"
$ProdConfig = "prod"
$LocalConfig = "dev_local"
$PostgresClientImage = "postgres:17"
# App-owned schemas only — skip Supabase internals (vault, auth, storage, …).
$AppSchemas = @("platform", "games", "other")
$SyncDir = Join-Path (Resolve-Path "$PSScriptRoot\..\..").Path "storage\sync"
$DumpFile = Join-Path $SyncDir "cloud.dump"

function Get-DopplerPlain {
    param(
        [Parameter(Mandatory = $true)][string]$Config,
        [Parameter(Mandatory = $true)][string]$Key
    )
    $value = doppler secrets get $Key --project $Project --config $Config --plain 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($value)) {
        throw "Missing Doppler secret '$Key' in config '$Config'."
    }
    return $value.Trim()
}

function Get-DopplerOptional {
    param(
        [Parameter(Mandatory = $true)][string]$Config,
        [Parameter(Mandatory = $true)][string]$Key
    )
    $value = doppler secrets get $Key --project $Project --config $Config --plain 2>$null
    if ($LASTEXITCODE -ne 0) {
        return ""
    }
    return $value.Trim()
}

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command not found on PATH: $Name"
    }
}

function Get-DockerPostgresHost {
    param([string]$HostName)
    if ($HostName -in @("localhost", "127.0.0.1")) {
        return "host.docker.internal"
    }
    return $HostName
}

function Invoke-PostgresClient {
    param(
        [Parameter(Mandatory = $true)][string[]]$ClientArgs,
        [hashtable]$EnvVars = @{}
    )
    $dockerArgs = @("run", "--rm")
    foreach ($key in $EnvVars.Keys) {
        $dockerArgs += @("-e", "${key}=$($EnvVars[$key])")
    }
    $dockerArgs += @(
        "-v", "${SyncDir}:/sync",
        $PostgresClientImage
    )
    $dockerArgs += $ClientArgs
    & docker --context desktop-linux @dockerArgs
    if ($LASTEXITCODE -ne 0) {
        throw "postgres client failed: $($ClientArgs -join ' ')"
    }
}

function Test-PostgresRole {
    param(
        [Parameter(Mandatory = $true)][string]$ComposeInfra,
        [Parameter(Mandatory = $true)][string]$User
    )
    $prevPref = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & docker --context desktop-linux compose -f $ComposeInfra exec -T postgres `
            psql -U $User -d postgres -tAc "SELECT 1" 2>&1 | Out-Null
        return ($LASTEXITCODE -eq 0)
    } finally {
        $ErrorActionPreference = $prevPref
    }
}

function Get-PostgresAdminUser {
    param(
        [Parameter(Mandatory = $true)][string]$ComposeInfra,
        [Parameter(Mandatory = $true)][string]$PreferredUser
    )
    $candidates = @()
    if (-not [string]::IsNullOrWhiteSpace($PreferredUser)) {
        $candidates += $PreferredUser
    }
    if ($PreferredUser -ne "postgres") {
        $candidates += "postgres"
    }
    foreach ($candidate in $candidates) {
        if (Test-PostgresRole -ComposeInfra $ComposeInfra -User $candidate) {
            return $candidate
        }
    }
    throw @"
Cannot connect to local Postgres. The data volume may be from an old init with different credentials.
Fix: make dev_down, remove storage/postgres, set POSTGRES_* in Doppler dev_local, then make dev_infra and retry.
"@
}

function Ensure-PostgresAppUser {
    param(
        [Parameter(Mandatory = $true)][string]$ComposeInfra,
        [Parameter(Mandatory = $true)][string]$AdminUser,
        [Parameter(Mandatory = $true)][string]$AppUser,
        [Parameter(Mandatory = $true)][string]$AppPassword
    )
    if ($AdminUser -eq $AppUser) {
        return
    }
    $escapedPassword = $AppPassword.Replace("'", "''")
    $prevPref = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $exists = (& docker --context desktop-linux compose -f $ComposeInfra exec -T postgres `
            psql -U $AdminUser -d postgres -tAc "SELECT 1 FROM pg_roles WHERE rolname = '$AppUser'" 2>&1).Trim()
    } finally {
        $ErrorActionPreference = $prevPref
    }
    if ($exists -ne "1") {
        & docker --context desktop-linux compose -f $ComposeInfra exec -T postgres `
            psql -U $AdminUser -d postgres -v ON_ERROR_STOP=1 -c `
            "CREATE ROLE $AppUser WITH LOGIN SUPERUSER CREATEDB PASSWORD '$escapedPassword';"
    } else {
        & docker --context desktop-linux compose -f $ComposeInfra exec -T postgres `
            psql -U $AdminUser -d postgres -v ON_ERROR_STOP=1 -c `
            "ALTER ROLE $AppUser WITH LOGIN SUPERUSER CREATEDB PASSWORD '$escapedPassword';"
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to ensure Postgres role '$AppUser'"
    }
}

Write-Host "Loading Doppler configs: source=$ProdConfig destination=$LocalConfig" -ForegroundColor Cyan

Require-Command "doppler"
Require-Command "docker"
Require-Command "aws"

$cloudHost = Get-DopplerPlain $ProdConfig "POSTGRES_HOST"
$cloudPort = Get-DopplerPlain $ProdConfig "POSTGRES_PORT"
$cloudDb = Get-DopplerPlain $ProdConfig "POSTGRES_DB"
$cloudUser = Get-DopplerPlain $ProdConfig "POSTGRES_USER"
$cloudPassword = Get-DopplerPlain $ProdConfig "POSTGRES_PASSWORD"
$cloudSsl = Get-DopplerOptional $ProdConfig "POSTGRES_SSLMODE"

$localHost = Get-DopplerPlain $LocalConfig "POSTGRES_HOST"
$localPort = Get-DopplerPlain $LocalConfig "POSTGRES_PORT"
$localDb = Get-DopplerPlain $LocalConfig "POSTGRES_DB"
$localUser = Get-DopplerPlain $LocalConfig "POSTGRES_USER"
$localPassword = Get-DopplerPlain $LocalConfig "POSTGRES_PASSWORD"

$cloudBucket = Get-DopplerPlain $ProdConfig "S3_BUCKET"
$cloudStorageRoot = Get-DopplerPlain $ProdConfig "STORAGE_ROOT"
$cloudS3Endpoint = Get-DopplerPlain $ProdConfig "S3_ENDPOINT_URL"
$cloudS3Key = Get-DopplerPlain $ProdConfig "S3_ACCESS_KEY_ID"
$cloudS3Secret = Get-DopplerPlain $ProdConfig "S3_SECRET_ACCESS_KEY"

$localBucket = Get-DopplerPlain $LocalConfig "S3_BUCKET"
$localStorageRoot = Get-DopplerPlain $LocalConfig "STORAGE_ROOT"
$localS3Endpoint = Get-DopplerPlain $LocalConfig "S3_ENDPOINT_URL"
$localS3Key = Get-DopplerPlain $LocalConfig "S3_ACCESS_KEY_ID"
$localS3Secret = Get-DopplerPlain $LocalConfig "S3_SECRET_ACCESS_KEY"

if (-not $Force) {
    Write-Host ""
    Write-Host "This will REPLACE local Postgres database '$localDb' and sync S3 prefix into MinIO." -ForegroundColor Yellow
    $confirm = Read-Host "Type 'yes' to continue"
    if ($confirm -ne "yes") {
        Write-Host "Aborted."
        exit 0
    }
}

New-Item -ItemType Directory -Force -Path $SyncDir | Out-Null

Write-Host ""
Write-Host "==> pg_dump from cloud Postgres ($PostgresClientImage container)" -ForegroundColor Cyan
$dumpEnv = @{ PGPASSWORD = $cloudPassword }
if ($cloudSsl) { $dumpEnv["PGSSLMODE"] = $cloudSsl }
$dumpArgs = @(
    "pg_dump", "--format=custom", "--no-owner", "--no-acl",
    "-h", $cloudHost, "-p", $cloudPort, "-U", $cloudUser, "-d", $cloudDb,
    "-f", "/sync/cloud.dump"
)
foreach ($schema in $AppSchemas) {
    $dumpArgs += "--schema=$schema"
}
Write-Host "Dumping schemas: $($AppSchemas -join ', ')" -ForegroundColor DarkGray
Invoke-PostgresClient -EnvVars $dumpEnv -ClientArgs $dumpArgs
Write-Host "Dump saved to $DumpFile"

Write-Host ""
Write-Host "==> Reset local Postgres (docker compose postgres service)" -ForegroundColor Cyan
$composeInfra = Join-Path (Resolve-Path "$PSScriptRoot\..\..").Path "docker-compose.infra.yml"
$adminUser = Get-PostgresAdminUser -ComposeInfra $composeInfra -PreferredUser $localUser
if ($adminUser -ne $localUser) {
    Write-Host "Local volume admin is '$adminUser' (Doppler user is '$localUser'); ensuring app role exists." -ForegroundColor Yellow
    Ensure-PostgresAppUser -ComposeInfra $composeInfra -AdminUser $adminUser -AppUser $localUser -AppPassword $localPassword
}
& docker --context desktop-linux compose -f $composeInfra exec -T postgres `
    dropdb -U $adminUser --if-exists $localDb
if ($LASTEXITCODE -ne 0) { throw "dropdb failed" }
& docker --context desktop-linux compose -f $composeInfra exec -T postgres `
    createdb -U $adminUser $localDb
if ($LASTEXITCODE -ne 0) { throw "createdb failed" }

Write-Host ""
Write-Host "==> pg_restore into local Postgres ($PostgresClientImage container)" -ForegroundColor Cyan
$restoreHost = Get-DockerPostgresHost $localHost
try {
    Invoke-PostgresClient -EnvVars @{ PGPASSWORD = $localPassword } -ClientArgs @(
        "pg_restore", "--no-owner", "--no-acl",
        "-h", $restoreHost, "-p", $localPort, "-U", $localUser, "-d", $localDb,
        "/sync/cloud.dump"
    )
} catch {
    Write-Host "pg_restore failed or reported warnings (extension mismatches are common on Supabase dumps)." -ForegroundColor Yellow
    Write-Host $_.Exception.Message -ForegroundColor Yellow
}

Write-Host ""
Write-Host "==> aws s3 sync cloud -> local MinIO" -ForegroundColor Cyan
$stagingDir = Join-Path $SyncDir "s3-staging"
if (Test-Path $stagingDir) {
    Remove-Item $stagingDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $stagingDir | Out-Null

$cloudPrefix = "s3://$cloudBucket/$cloudStorageRoot/"
$localPrefix = "s3://$localBucket/$localStorageRoot/"

$env:AWS_ACCESS_KEY_ID = $cloudS3Key
$env:AWS_SECRET_ACCESS_KEY = $cloudS3Secret
& aws s3 sync $cloudPrefix $stagingDir --endpoint-url $cloudS3Endpoint
if ($LASTEXITCODE -ne 0) { throw "aws s3 sync (cloud download) failed" }

$env:AWS_ACCESS_KEY_ID = $localS3Key
$env:AWS_SECRET_ACCESS_KEY = $localS3Secret
& aws s3 sync $stagingDir $localPrefix --endpoint-url $localS3Endpoint
if ($LASTEXITCODE -ne 0) { throw "aws s3 sync (local MinIO upload) failed" }

Write-Host ""
Write-Host "Cloud -> local sync complete." -ForegroundColor Green
