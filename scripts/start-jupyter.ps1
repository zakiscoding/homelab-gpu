# Start JupyterLab for fine-tuning notebooks (GPU PC — zakar)
# Default: http://127.0.0.1:8888 — use Tailscale IP from laptop if firewall allows 8888
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Error "Run .\scripts\setup-venv.ps1 first"
}

# Load .env into process (optional)
if (Test-Path ".\.env") {
    Get-Content ".\.env" | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
        }
    }
}

$port = if ($env:JUPYTER_PORT) { $env:JUPYTER_PORT } else { "8888" }
$ip = if ($env:JUPYTER_LISTEN) { $env:JUPYTER_LISTEN } else { "127.0.0.1" }

Write-Host "JupyterLab: http://${ip}:${port}"
Write-Host "Stop Ollama first if training: .\scripts\stop-ollama.ps1"
Write-Host ""

.\.venv\Scripts\python.exe -m jupyter lab --ip=$ip --port=$port --no-browser
