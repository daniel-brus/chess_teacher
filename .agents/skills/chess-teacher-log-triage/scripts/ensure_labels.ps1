# Create GitHub labels for triage/implement (idempotent).
$ErrorActionPreference = "Continue"
$labels = @(
    @{ Name = "from-logs"; Color = "0E8A16"; Description = "Opened from S3/app log triage" },
    @{ Name = "from-vps"; Color = "1D76DB"; Description = "Opened from VPS kubectl log triage" },
    @{ Name = "from-ci"; Color = "5319E7"; Description = "Opened from GitHub Actions triage" },
    @{ Name = "agent-fix"; Color = "D93F0B"; Description = "Ready for Implement Automation" },
    @{ Name = "in-progress"; Color = "FBCA04"; Description = "Implement Automation claimed" },
    @{ Name = "pr-open"; Color = "C2E0C6"; Description = "Fix PR opened" },
    @{ Name = "wontfix"; Color = "ffffff"; Description = "Do not auto-reopen" }
)
foreach ($l in $labels) {
    gh label create $l.Name --color $l.Color --description $l.Description 2>$null
    if ($LASTEXITCODE -ne 0) {
        gh label edit $l.Name --color $l.Color --description $l.Description
    }
}
Write-Output "labels ok"
