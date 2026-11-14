# Re-register homelab tasks so nothing flashes a console (run as Administrator).
#Requires -RunAsAdministrator
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$Vbs = Join-Path $Repo "scripts\run-hidden-powershell.vbs"
$MetricsPs1 = Join-Path $Repo "scripts\push-gpu-metrics.ps1"
$Ollama = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
if (-not (Test-Path $Ollama)) {
  $cmd = Get-Command ollama -ErrorAction SilentlyContinue
  if ($cmd) { $Ollama = $cmd.Source } else { throw "Ollama not found" }
}

function Register-HiddenTask {
  param(
    [string]$TaskName,
    [string]$Execute,
    [string]$Arguments,
    [CimInstance[]]$Trigger,
    [string]$Description = ""
  )
  $action = New-ScheduledTaskAction -Execute $Execute -Argument $Arguments
  $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -Hidden
  $params = @{
    TaskName  = $TaskName
    Action    = $action
    Trigger   = $Trigger
    Settings  = $settings
    Force     = $true
  }
  if ($Description) { $params.Description = $Description }
  Register-ScheduledTask @params | Out-Null
  Write-Host "OK: $TaskName -> $Execute"
}

# Metrics: wscript launcher (Hidden flag alone is not enough for powershell.exe)
$metricsTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
  -RepetitionInterval (New-TimeSpan -Minutes 1) -RepetitionDuration (New-TimeSpan -Days 3650)
Register-HiddenTask -TaskName "HomelabGpuMetrics" `
  -Execute "wscript.exe" `
  -Arguments "//B `"$Vbs`" `"$MetricsPs1`"" `
  -Trigger $metricsTrigger `
  -Description "Push GPU metrics to iMac (no console)"

# Ollama: never use .bat - it leaves a visible cmd window
$ollamaTrigger = (Get-ScheduledTask -TaskName "HomelabOllamaServe" -ErrorAction SilentlyContinue).Triggers
if (-not $ollamaTrigger) {
  $ollamaTrigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
}
Register-HiddenTask -TaskName "HomelabOllamaServe" `
  -Execute $Ollama `
  -Arguments "serve" `
  -Trigger $ollamaTrigger `
  -Description "Ollama serve 0.0.0.0:11434 (no console)"

# Legacy duplicate task - hide or disable if present
foreach ($legacy in @("OllamaServe")) {
  $t = Get-ScheduledTask -TaskName $legacy -ErrorAction SilentlyContinue
  if ($t) {
    Disable-ScheduledTask -TaskName $legacy | Out-Null
    Write-Host "Disabled duplicate: $legacy (use HomelabOllamaServe)"
  }
}

Write-Host ""
Write-Host "Verify:"
Get-ScheduledTask -TaskName "HomelabGpuMetrics","HomelabOllamaServe" |
  ForEach-Object { "  $($_.TaskName): $($_.Actions.Execute) $($_.Actions.Arguments)" }
