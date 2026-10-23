# Reaction CNN HTTP service (for homelab-web / iMac API proxy)
$ErrorActionPreference = "Stop"
$Repo = Split-Path (Split-Path $MyInvocation.MyCommand.Path -Parent) -Parent
$Venv = Join-Path $Repo ".venv\Scripts"
if (-not $env:REACTION_CHECKPOINT) {
    $env:REACTION_CHECKPOINT = "C:\Users\zakar\homelab-data\checkpoints\8117bfe1-85ea-4074-a967-5ebeedd40698\best_model.pt"
}
$port = if ($env:VISION_SERVE_PORT) { $env:VISION_SERVE_PORT } else { "8100" }
Write-Host "Reaction serve on :$port checkpoint=$env:REACTION_CHECKPOINT"
& "$Venv\python.exe" -m pip install -q fastapi "uvicorn[standard]" opencv-python-headless 2>$null
Set-Location $Repo
& "$Venv\uvicorn.exe" reaction.serve:app --host 0.0.0.0 --port $port
