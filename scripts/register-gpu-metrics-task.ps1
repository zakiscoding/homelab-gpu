# Scheduled task: push GPU metrics every 30s to iMac Pushgateway.
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$Script = Join-Path $Repo "scripts\push-gpu-metrics.ps1"
$TaskName = "HomelabGpuMetrics"

$vbs = Join-Path $Repo "scripts\run-hidden-powershell.vbs"
$action = New-ScheduledTaskAction -Execute "wscript.exe" `
    -Argument "//B `"$vbs`" `"$Script`""
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes 1) -RepetitionDuration (New-TimeSpan -Days 3650)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -Hidden

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Force | Out-Null
Write-Output "Registered $TaskName (hidden, repeats every minute)"
