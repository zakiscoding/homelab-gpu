# Prefer fix-console-popups.ps1 (Admin) — Hidden alone does not stop powershell flashes.
# This only sets the Hidden flag on existing tasks.
$ErrorActionPreference = "Stop"
Write-Warning "If consoles still flash, run as Admin: .\scripts\fix-console-popups.ps1"

$names = @("HomelabGpuMetrics", "HomelabGpuTrainWorker", "HomelabOllamaServe", "OllamaServe")
foreach ($name in $names) {
    $task = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
    if (-not $task) { continue }
    $task.Settings.Hidden = $true
    $task | Set-ScheduledTask | Out-Null
    Write-Output "Hidden flag set: $name"
}
