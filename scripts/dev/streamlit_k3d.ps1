# Free port 8501 if an old kubectl port-forward is listening, then open a tunnel
# window to the k3d Streamlit service.
$ErrorActionPreference = "Continue"

$listeners = Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue
foreach ($conn in $listeners) {
    if ($conn.OwningProcess) {
        Write-Host "Freeing port 8501 (PID $($conn.OwningProcess))..." -ForegroundColor DarkGray
        Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "Starting port-forward http://127.0.0.1:8501 -> svc/streamlit ..." -ForegroundColor Cyan
Start-Process -FilePath "cmd.exe" -ArgumentList @(
    "/k",
    "kubectl port-forward --address 127.0.0.1 -n chess-teacher svc/streamlit 8501:8501"
) -WindowStyle Normal

Write-Host "Open http://localhost:8501 (close the port-forward window to stop the tunnel)." -ForegroundColor Green
exit 0
