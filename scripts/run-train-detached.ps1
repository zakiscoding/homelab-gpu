# Run QLoRA train for one job_id (detached). Usage: .\run-train-detached.ps1 -JobId <uuid>
param(
    [Parameter(Mandatory = $true)]
    [string]$JobId
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$LogDir = "C:\Users\zakar\homelab-data"
$Log = Join-Path $LogDir "train-last.log"
$Err = Join-Path $LogDir "train-last.err"
$Bat = Join-Path $LogDir "train-run-once.bat"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$py = Join-Path $Repo ".venv\Scripts\python.exe"
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
$lines += 'set "TRAIN_MODE=qlora"'
$lines += "`"$py`" -u scripts\train_once_job.py $JobId 1>> `"$Log`" 2>> `"$Err`""
$lines | Set-Content -Path $Bat -Encoding ASCII

$proc = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "cmd.exe /c `"$Bat`"" }
if ($proc.ReturnValue -ne 0) { throw "Win32_Process.Create failed: $($proc.ReturnValue)" }
Write-Output "train pid=$($proc.ProcessId) log=$Log err=$Err"
