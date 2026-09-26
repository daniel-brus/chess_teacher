# Phase 3a overnight — balanced 50/50 parent then user FT
#
# Run from repo root (PowerShell). Expect multi-hour wall (pack dominates).
# Logs + artifacts under storage/tmp/phase3a/
#
# Note: Python/doppler write INFO to stderr; do NOT use $ErrorActionPreference=Stop
# around those calls or PowerShell treats stderr as a terminating NativeCommandError.

Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
New-Item -ItemType Directory -Force -Path storage/tmp/phase3a | Out-Null

$Daniel = "5c3f069730e098aedc61b978c92d6448eebfdf234e575552e86343a10bfb698b"
$Rebecca = "d984e60e96a24fc53262d2579bf80cf9fb31290d806869a1d567dcb12e5e0c01"
$Parent = "storage/tmp/phase3a/hybrid_parent_25k25k.keras"
$Py = ".venv\Scripts\python.exe"

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$ArgList,
        [string]$Log,
        [int[]]$OkExit = @(0)
    )
    Write-Host "=== START $Name $(Get-Date -Format o) ==="
    # cmd.exe redirect avoids PS NativeCommandError on stderr INFO lines
    $argStr = ($ArgList | ForEach-Object {
        if ($_ -match '[\s,]') { '"' + $_ + '"' } else { $_ }
    }) -join ' '
    $cmd = "doppler run --project chess-teacher --config dev_local -- $Py $argStr > `"$Log`" 2>&1"
    cmd /c $cmd
    $code = $LASTEXITCODE
    Write-Host "=== END $Name exit=$code $(Get-Date -Format o) ==="
    if ($OkExit -notcontains $code) {
        throw "Step $Name failed exit=$code (log=$Log)"
    }
    return $code
}

$skipParent = $env:PHASE3A_SKIP_PARENT -eq "1"
if (-not $skipParent) {
    Invoke-Step "parent_25k25k" @(
        "scripts/tools/offline_user_finetune_eval.py",
        "--train-parent",
        "--parent-out", $Parent,
        "--parent-train-limit", "25000",
        "--parent-balance-account-ids", "$Daniel,$Rebecca",
        "--parent-epochs", "20",
        "--parent-style-disagree-boost", "2.0",
        "--split-version", "baseline-v1"
    ) "storage/tmp/phase3a/parent_25k25k_run.log"
}

if (-not (Test-Path $Parent)) { throw "Parent keras missing: $Parent" }

Invoke-Step "ikbendaniel_ft" @(
    "scripts/tools/offline_user_finetune_eval.py",
    "--account-id", $Daniel,
    "--parent-weights", $Parent,
    "--child-out", "storage/tmp/phase3a/ikbendaniel_ft_30k.keras",
    "--train-limit", "30000",
    "--val-limit", "8000",
    "--epochs", "20",
    "--style-disagree-boost", "4.0",
    "--split-version", "baseline-v1"
) "storage/tmp/phase3a/ikbendaniel_ft_30k_run.log"

Invoke-Step "RebeccaHarris_ft" @(
    "scripts/tools/offline_user_finetune_eval.py",
    "--account-id", $Rebecca,
    "--parent-weights", $Parent,
    "--child-out", "storage/tmp/phase3a/RebeccaHarris_ft_30k.keras",
    "--train-limit", "30000",
    "--val-limit", "8000",
    "--epochs", "20",
    "--style-disagree-boost", "4.0",
    "--min-train-moves", "300",
    "--split-version", "baseline-v1"
) "storage/tmp/phase3a/RebeccaHarris_ft_30k_run.log" -OkExit @(0, 2)

Write-Host "=== overnight chain complete $(Get-Date -Format o) ==="
"done $(Get-Date -Format o)" | Set-Content storage/tmp/phase3a/OVERNIGHT_DONE.txt
