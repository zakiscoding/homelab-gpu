# Run one poll cycle (debug). For daemon use start-gpu-worker.ps1 with .env loaded.
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

if (Test-Path "$Repo\.env") {
    Get-Content "$Repo\.env" | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), 'Process')
        }
    }
}

if (-not $env:WORKER_API_KEY) { throw "WORKER_API_KEY not set (use .env)" }
if (-not $env:API_BASE_URL) { $env:API_BASE_URL = "http://100.91.113.98" }

& "$Repo\.venv\Scripts\python.exe" -m worker.poll_jobs --once
