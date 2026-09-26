# Wait for current scale100k chain to finish (Rebecca ladder), then resume
# ikbendaniel r6+ with OOM fix applied. Does not touch the enrich k8s job.
$ErrorActionPreference = "Continue"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $Root
$Scale = "storage/tmp/phase3a/scale100k"
$Done = Join-Path $Scale "DONE.txt"
$Status = Join-Path $Scale "STATUS.json"
$Outer = Join-Path $Scale "daniel_resume_outer.log"

Write-Output "$(Get-Date -Format o) watcher start; wait for Rebecca chain DONE"

while (-not (Test-Path $Done)) {
    $rh8 = $false
    if (Test-Path $Status) {
        $blob = Get-Content $Status -Raw | ConvertFrom-Json
        foreach ($s in $blob.steps) {
            if ($s.step -eq "ft_RebeccaHarris_r8" -and $s.state -eq "done" -and $s.registered) {
                $rh8 = $true
            }
        }
    }
    # Chain may still be writing DONE after r8; also exit wait if shell died with r8 done.
    if ($rh8 -and (Test-Path $Done)) { break }
    if ($rh8) {
        Start-Sleep -Seconds 30
        if (Test-Path $Done) { break }
        # r8 done+registered but no DONE yet: chain likely finishing; wait a bit more
        Start-Sleep -Seconds 60
        break
    }
    Start-Sleep -Seconds 120
}

Write-Output "$(Get-Date -Format o) Rebecca ladder settled; starting ikbendaniel r6+ resume"
cmd /c "doppler run --project chess-teacher --config dev_local -- .venv\Scripts\python.exe scripts/tools/phase3a_scale_100k_chain.py --max-rounds 8 --skip-parent --accounts ikbendaniel > $Outer 2>&1"
Write-Output "EXIT=$LASTEXITCODE"
Get-Content $Outer -Tail 30 -EA SilentlyContinue
