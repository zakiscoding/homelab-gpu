# Run once on GPU PC as Administrator — keeps Ollama on 0.0.0.0:11434 across reboots.
#Requires -RunAsAdministrator
$ErrorActionPreference = "Stop"

$hostAddr = "0.0.0.0:11434"
$ollama = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
if (-not (Test-Path $ollama)) {
  $cmd = Get-Command ollama -ErrorAction SilentlyContinue
  if ($cmd) { $ollama = $cmd.Source } else { throw "Install Ollama first" }
}

[System.Environment]::SetEnvironmentVariable("OLLAMA_HOST", $hostAddr, "Machine")

$taskName = "HomelabOllamaServe"
$action = New-ScheduledTaskAction -Execute $ollama -Argument "serve" `
  -WorkingDirectory (Split-Path $ollama)
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
  -RestartInterval (New-TimeSpan -Minutes 1) -RestartCount 999 -Hidden
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
  -Settings $settings -Principal $principal -Force | Out-Null

$existing = Get-NetFirewallRule -DisplayName "Ollama LAN" -ErrorAction SilentlyContinue
if (-not $existing) {
  New-NetFirewallRule -DisplayName "Ollama LAN" -Direction Inbound -Protocol TCP -LocalPort 11434 `
    -Action Allow -Profile Any | Out-Null
}

Stop-Process -Name ollama* -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
Start-ScheduledTask -TaskName $taskName
Start-Sleep -Seconds 8

$listen = Get-NetTCPConnection -LocalPort 11434 -State Listen -ErrorAction SilentlyContinue
if (-not $listen) { throw "Ollama not listening — check Task Scheduler > $taskName" }

Write-Host "OK: Ollama on port 11434"
Write-Host "Listen:" (($listen | ForEach-Object { $_.LocalAddress }) -join ", ")
Invoke-RestMethod http://127.0.0.1:11434/api/tags -TimeoutSec 10 | Out-Null
Write-Host "Test from laptop: curl.exe http://100.88.71.33:11434/api/tags"
