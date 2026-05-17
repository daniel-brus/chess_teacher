$scriptPath = Join-Path $PSScriptRoot "nightly_dev_job.ps1"

$action = New-ScheduledTaskAction `
  -Execute "powershell.exe" `
  -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`""
$trigger = New-ScheduledTaskTrigger -Daily -At 3:00AM

$settings = New-ScheduledTaskSettingsSet `
  -WakeToRun `
  -StartWhenAvailable `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries

Register-ScheduledTask `
  -TaskName "ChessTeacherNightlyDev" `
  -Action $action `
  -Trigger $trigger `
  -Settings $settings `
  -Force
