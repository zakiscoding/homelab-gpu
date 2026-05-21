# One-time: create venv and install base deps (run from repo root)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python not found. Install: winget install Python.Python.3.12"
}

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Write-Host "Done. Activate with: .\.venv\Scripts\Activate.ps1"
