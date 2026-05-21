# Run one worker cycle detached from SSH session (survives disconnect).
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$LogDir = "C:\Users\zakar\homelab-data"
$Log = Join-Path $LogDir "worker-last.log"
$Err = Join-Path $LogDir "worker-last.err"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if (Test-Path "$Repo\.env") {
    Get-Content "$Repo\.env" | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), 'Process')
        }
    }
}
if (-not $env:TRAIN_MODE) { $env:TRAIN_MODE = "qlora" }

$py = Join-Path $Repo ".venv\Scripts\python.exe"
$args = @("-m", "worker.poll_jobs", "--once")
$p = Start-Process -FilePath $py -ArgumentList $args -WorkingDirectory $Repo `
    -WindowStyle Hidden -PassThru -RedirectStandardOutput $Log -RedirectStandardError $Err
Write-Output "started pid=$($p.Id) log=$Log"
