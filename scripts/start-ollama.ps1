# Start Ollama on 0.0.0.0:11434 (Tailscale + LAN). Run as Administrator for firewall rule.
$ErrorActionPreference = "Stop"
$hostAddr = "0.0.0.0:11434"

[System.Environment]::SetEnvironmentVariable("OLLAMA_HOST", $hostAddr, "Machine")
$env:OLLAMA_HOST = $hostAddr

$ollama = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
if (-not (Test-Path $ollama)) {
  $cmd = Get-Command ollama -ErrorAction SilentlyContinue
  if ($cmd) { $ollama = $cmd.Source } else { throw "ollama.exe not found" }
}

Stop-Process -Name ollama* -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3

# Child inherits this process env (Machine OLLAMA_HOST alone is not enough on Windows).
Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden
Start-Sleep -Seconds 6

$listen = Get-NetTCPConnection -LocalPort 11434 -State Listen -ErrorAction SilentlyContinue
if (-not $listen) { throw "Ollama not listening on 11434" }
Write-Host "Listen:" (($listen | ForEach-Object { $_.LocalAddress }) -join ", ")

if (-not (Get-NetFirewallRule -DisplayName "Ollama LAN" -ErrorAction SilentlyContinue)) {
  New-NetFirewallRule -DisplayName "Ollama LAN" -Direction Inbound -Protocol TCP -LocalPort 11434 `
    -Action Allow -Profile Any | Out-Null
  Write-Host "Firewall rule added (run as Admin if this failed)."
}

$tags = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 15
Write-Host "Models:" ($tags.models.name -join ", ")
