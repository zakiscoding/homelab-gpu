# Run QLoRA train for one job_id (detached). Usage: .\run-train-detached.ps1 -JobId <uuid>
param(
    [Parameter(Mandatory = $true)]
    [string]$JobId,
    [string]$DatasetVersion = "smoke",
    [string]$ModelVersion = "phase3-smoke-v1"
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$LogDir = "C:\Users\zakar\homelab-data"
$Log = Join-Path $LogDir "train-last.log"
$Err = Join-Path $LogDir "train-last.err"
$Bat = Join-Path $LogDir "train-run-once.bat"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$payload = "{`"dataset_version`":`"$DatasetVersion`",`"model_version`":`"$ModelVersion`"}"
$py = Join-Path $Repo ".venv\Scripts\python.exe"
$code = @"
import json, os, sys
sys.path.insert(0, r'$Repo')
from worker.run_train import run_train
p = json.loads(r'''$payload''')
root, m = run_train(r'$JobId', p)
print('done', root, m)
"@

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
$lines += "`"$py`" -u -c `"$($code -replace '"','\"')`" 1>> `"$Log`" 2>> `"$Err`""
$lines | Set-Content -Path $Bat -Encoding ASCII

$proc = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "cmd.exe /c `"$Bat`"" }
if ($proc.ReturnValue -ne 0) { throw "Win32_Process.Create failed: $($proc.ReturnValue)" }
Write-Output "train pid=$($proc.ProcessId) log=$Log"
