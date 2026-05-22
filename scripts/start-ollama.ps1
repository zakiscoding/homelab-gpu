# Quick start (session). For always-on use install-ollama-serve-task.ps1 as Admin.
$ErrorActionPreference = "Stop"
$hostAddr = "0.0.0.0:11434"
$env:OLLAMA_HOST = $hostAddr

$ollama = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
if (-not (Test-Path $ollama)) {
  $cmd = Get-Command ollama -ErrorAction SilentlyContinue
  if ($cmd) { $ollama = $cmd.Source } else { throw "ollama.exe not found" }
}

$task = Get-ScheduledTask -TaskName "HomelabOllamaServe" -ErrorAction SilentlyContinue
if ($task) {
  Start-ScheduledTask -TaskName "HomelabOllamaServe"
  Start-Sleep -Seconds 6
} else {
  Stop-Process -Name ollama* -Force -ErrorAction SilentlyContinue
  Start-Sleep -Seconds 2
  Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden
  Start-Sleep -Seconds 6
}

$listen = Get-NetTCPConnection -LocalPort 11434 -State Listen -ErrorAction SilentlyContinue
if (-not $listen) {
  Write-Host "Not listening. Run as Administrator:"
  Write-Host "  .\scripts\install-ollama-serve-task.ps1"
  exit 1
}
Write-Host "Listen:" (($listen | ForEach-Object { $_.LocalAddress }) -join ", ")
$tags = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 15
Write-Host "Models:" ($tags.models.name -join ", ")
