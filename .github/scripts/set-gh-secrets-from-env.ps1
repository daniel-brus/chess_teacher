param(
  [Parameter(Mandatory = $false)]
  [string]$EnvPath = ".\.env",

  [Parameter(Mandatory = $false)]
  [string[]]$Keys = @(),

  [Parameter(Mandatory = $false)]
  [string]$EnvironmentName = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $EnvPath)) {
  throw "Env file not found: $EnvPath"
}

function Parse-DotEnv([string]$path) {
  $result = @{}
  foreach ($rawLine in Get-Content -LiteralPath $path) {
    $line = $rawLine.Trim()
    if (-not $line) { continue }
    if ($line.StartsWith("#")) { continue }
    if ($line -notmatch "=") { continue }

    $parts = $line.Split("=", 2)
    $key = $parts[0].Trim()
    $value = $parts[1]

    if ($key -and -not $result.ContainsKey($key)) {
      $result[$key] = $value
    }
  }
  return $result
}

$dotEnv = Parse-DotEnv $EnvPath

$selectedKeys = $Keys
if (-not $selectedKeys -or $selectedKeys.Count -eq 0) {
  $selectedKeys = @($dotEnv.Keys)
}

foreach ($key in $selectedKeys) {
  if (-not $dotEnv.ContainsKey($key)) {
    Write-Warning "Key not found in .env: $key"
    continue
  }

  $value = $dotEnv[$key]
  if ($EnvironmentName) {
    $value | gh secret set $key --env $EnvironmentName -f -
  } else {
    $value | gh secret set $key -f -
  }
}

Write-Output "Done."
