# Run one worker cycle in a process tree that survives SSH disconnect.
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$LogDir = "C:\Users\zakar\homelab-data"
$Log = Join-Path $LogDir "worker-last.log"
$Err = Join-Path $LogDir "worker-last.err"
$Bat = Join-Path $LogDir "worker-run-once.bat"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$lines = @("@echo off", "cd /d `"$Repo`"")
if (Test-Path "$Repo\.env") {
    Get-Content "$Repo\.env" | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            $k = $matches[1].Trim()
            $v = $matches[2].Trim().Trim('"')
            $lines += "set `"$k=$v`""
        }
    }
}
if (-not $env:TRAIN_MODE) { $lines += 'set "TRAIN_MODE=qlora"' } else { $lines += "set `"TRAIN_MODE=$($env:TRAIN_MODE)`"" }

$py = Join-Path $Repo ".venv\Scripts\python.exe"
$lines += "`"$py`" -u -m worker.poll_jobs --once 1>> `"$Log`" 2>> `"$Err`""
$lines | Set-Content -Path $Bat -Encoding ASCII

$proc = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "cmd.exe /c `"$Bat`"" }
if ($proc.ReturnValue -ne 0) { throw "Win32_Process.Create failed: $($proc.ReturnValue)" }
Write-Output "started pid=$($proc.ProcessId) bat=$Bat log=$Log"
