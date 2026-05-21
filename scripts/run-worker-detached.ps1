# Run one worker cycle in a process tree that survives SSH disconnect.
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$LogDir = "C:\Users\zakar\homelab-data"
$Log = Join-Path $LogDir "worker-last.log"
$Err = Join-Path $LogDir "worker-last.err"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$sets = @("cd /d `"$Repo`"")
if (Test-Path "$Repo\.env") {
    Get-Content "$Repo\.env" | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            $k = $matches[1].Trim()
            $v = $matches[2].Trim().Trim('"')
            $sets += "set $k=$v"
        }
    }
}
if (-not $env:TRAIN_MODE) { $sets += "set TRAIN_MODE=qlora" } else { $sets += "set TRAIN_MODE=$($env:TRAIN_MODE)" }

$py = Join-Path $Repo ".venv\Scripts\python.exe"
$sets += "`"$py`" -u -m worker.poll_jobs --once 1>> `"$Log`" 2>> `"$Err`""
$cmd = ($sets -join " && ")

$proc = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "cmd.exe /c $cmd" }
if ($proc.ReturnValue -ne 0) { throw "Win32_Process.Create failed: $($proc.ReturnValue)" }
Write-Output "started pid=$($proc.ProcessId) log=$Log err=$Err"
