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
if (-not $env:WORKER_API_KEY) { throw "WORKER_API_KEY not set (use .env)" }
$py = Join-Path $Repo ".venv\Scripts\python.exe"
$p = Start-Process -FilePath $py -ArgumentList "-m", "worker.poll_jobs" -WorkingDirectory $Repo -WindowStyle Hidden -PassThru
Write-Output "started pid=$($p.Id)"
