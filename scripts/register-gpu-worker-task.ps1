# Register Phase 1 GPU train worker as a Windows scheduled task (runs at logon).
# Requires admin once: Run PowerShell as Administrator.
$ErrorActionPreference = "Stop"
$Repo = "C:\Users\zakar\homelab-gpu"
$Script = Join-Path $Repo "scripts\start-gpu-worker-bg.ps1"
$TaskName = "HomelabGpuTrainWorker"

if (-not (Test-Path $Script)) {
    throw "Missing $Script — git pull homelab-gpu first"
}

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$Script`"" `
    -WorkingDirectory $Repo
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Description "Poll iMac API for train jobs (homelab Phase 0/1)" -Force

Write-Host "Registered scheduled task: $TaskName"
Write-Host "Start now: Start-ScheduledTask -TaskName $TaskName"
Write-Host "Status:    Get-ScheduledTask -TaskName $TaskName"
