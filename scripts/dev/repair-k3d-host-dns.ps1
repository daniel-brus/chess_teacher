# Laptop/k3d only. Re-inject host.k3d.internal into CoreDNS after k3s rewrites it
# (k3d issues 926/1112/1221). Not used on the production VPS.
param(
    [string]$ClusterName = "chess-teacher"
)

$ErrorActionPreference = "Stop"
$dockerContext = "desktop-linux"
$nodeName = "k3d-$ClusterName-server-0"
$networkName = "k3d-$ClusterName"

function Get-K3dHostIp {
    $raw = & docker --context $dockerContext exec $nodeName nslookup host.docker.internal 2>$null | Out-String
    $match = [regex]::Match($raw, '(?m)^Address:\s+(\d+\.\d+\.\d+\.\d+)\s*$')
    if ($match.Success) {
        return $match.Groups[1].Value
    }

    $gateway = & docker --context $dockerContext network inspect $networkName --format '{{(index .IPAM.Config 0).Gateway}}'
    if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($gateway)) {
        return $gateway.Trim()
    }

    throw "Cannot determine host IP for host.k3d.internal (nslookup host.docker.internal failed and no k3d network gateway)."
}

function Get-NodeHosts {
    $value = & kubectl -n kube-system get configmap coredns -o jsonpath='{.data.NodeHosts}'
    if ($LASTEXITCODE -ne 0) {
        throw "kubectl failed: get configmap coredns"
    }
    return [string]$value
}

$hostIp = Get-K3dHostIp
Write-Host "Repairing host.k3d.internal -> $hostIp" -ForegroundColor Cyan

$nodeHosts = Get-NodeHosts
$lines = @($nodeHosts -split "`r?`n" | Where-Object { $_ -and ($_ -notmatch '\shost\.k3d\.internal$') })
$lines += "$hostIp host.k3d.internal"
$newHosts = ($lines -join "`n") + "`n"

$patch = @{ data = @{ NodeHosts = $newHosts } } | ConvertTo-Json -Compress
$patchFile = Join-Path ([System.IO.Path]::GetTempPath()) "chess-teacher-coredns-nodehosts.json"
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
try {
    [System.IO.File]::WriteAllText($patchFile, $patch, $utf8NoBom)
    & kubectl -n kube-system patch configmap coredns --type merge --patch-file $patchFile
    if ($LASTEXITCODE -ne 0) {
        throw "kubectl failed: patch configmap coredns NodeHosts"
    }
} finally {
    if (Test-Path $patchFile) {
        Remove-Item $patchFile -Force
    }
}

$custom = @"
apiVersion: v1
kind: ConfigMap
metadata:
  name: coredns-custom
  namespace: kube-system
data:
  host-k3d.server: |
    host.k3d.internal:53 {
      errors
      hosts {
        $hostIp host.k3d.internal
      }
      cache 30
    }
"@
$custom | & kubectl apply -f -
if ($LASTEXITCODE -ne 0) {
    throw "kubectl failed: apply coredns-custom"
}

& kubectl -n kube-system rollout restart deployment/coredns
if ($LASTEXITCODE -ne 0) {
    throw "kubectl failed: rollout restart coredns"
}
& kubectl -n kube-system rollout status deployment/coredns --timeout=90s
if ($LASTEXITCODE -ne 0) {
    throw "kubectl failed: rollout status coredns"
}

Write-Host "CoreDNS answers host.k3d.internal ($hostIp)." -ForegroundColor Green
