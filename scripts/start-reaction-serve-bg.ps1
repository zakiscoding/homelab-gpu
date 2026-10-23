# Start reaction CNN serve detached (port 8100)
$Repo = "C:\Users\zakar\homelab-gpu"
$logDir = "C:\Users\zakar\homelab-data\logs"
$Py = "$Repo\.venv\Scripts\python.exe"
$Uvicorn = "$Repo\.venv\Scripts\uvicorn.exe"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

if (-not $env:REACTION_CHECKPOINT) {
    $env:REACTION_CHECKPOINT = "C:\Users\zakar\homelab-data\checkpoints\8117bfe1-85ea-4074-a967-5ebeedd40698\best_model.pt"
}
$port = if ($env:VISION_SERVE_PORT) { $env:VISION_SERVE_PORT } else { "8100" }

Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 1

$args = @("-m", "uvicorn", "reaction.serve:app", "--host", "0.0.0.0", "--port", $port)
Start-Process -FilePath $Py -ArgumentList $args -WorkingDirectory $Repo `
    -RedirectStandardOutput "$logDir\reaction-serve.log" `
    -RedirectStandardError "$logDir\reaction-serve.err" -WindowStyle Hidden

Write-Host "Reaction serve starting on :$port"
Write-Host "Checkpoint: $($env:REACTION_CHECKPOINT)"
Write-Host "Log: $logDir\reaction-serve.log"
