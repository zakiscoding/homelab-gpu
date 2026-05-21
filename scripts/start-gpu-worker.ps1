# Start Phase 0A GPU job worker (poll iMac API for train jobs).
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

$envFile = Join-Path $Repo ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), 'Process')
        }
    }
}

if (-not $env:WORKER_API_KEY) {
    Write-Error "Set WORKER_API_KEY in .env (must match iMac homelab-services .env)"
}

if (-not $env:API_BASE_URL) {
    $env:API_BASE_URL = "http://100.91.113.98"
}

$venvPython = Join-Path $Repo ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "Run scripts\setup-venv.ps1 first"
}

Write-Host "Starting GPU worker -> $env:API_BASE_URL"
& $venvPython -m worker.poll_jobs
