# Phase 3a — scale balanced parent only (leave enrich job alone).
# 50k ikbendaniel + 50k RebeccaHarris registry-train, ep20, boost 2.0
#
#   powershell -ExecutionPolicy Bypass -File scripts/tools/phase3a_parent_50k50k.ps1

Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
New-Item -ItemType Directory -Force -Path storage/tmp/phase3a | Out-Null

$Daniel = "5c3f069730e098aedc61b978c92d6448eebfdf234e575552e86343a10bfb698b"
$Rebecca = "d984e60e96a24fc53262d2579bf80cf9fb31290d806869a1d567dcb12e5e0c01"
$Parent = "storage/tmp/phase3a/hybrid_parent_50k50k.keras"
$Log = "storage/tmp/phase3a/parent_50k50k_run.log"
$Py = ".venv\Scripts\python.exe"

Write-Host "=== START parent_50k50k $(Get-Date -Format o) ==="
$argStr = @(
    "scripts/tools/offline_user_finetune_eval.py",
    "--train-parent",
    "--parent-out", $Parent,
    "--parent-train-limit", "50000",
    "--parent-balance-account-ids", "$Daniel,$Rebecca",
    "--parent-epochs", "20",
    "--parent-style-disagree-boost", "2.0",
    "--split-version", "baseline-v1"
) -join ' '
cmd /c "doppler run --project chess-teacher --config dev_local -- $Py $argStr > `"$Log`" 2>&1"
$code = $LASTEXITCODE
Write-Host "=== END parent_50k50k exit=$code $(Get-Date -Format o) ==="
if ($code -ne 0) { throw "parent_50k50k failed exit=$code log=$Log" }
if (-not (Test-Path $Parent)) { throw "missing $Parent" }
"parent_50k50k done $(Get-Date -Format o)" | Set-Content storage/tmp/phase3a/PARENT_50k50k_DONE.txt
Write-Host "Done. Next: register with register_phase3a_playables (extend) or FT children."
