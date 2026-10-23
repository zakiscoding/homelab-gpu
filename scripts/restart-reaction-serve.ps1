# Stop anything on 8100 and start reaction serve with latest or env checkpoint
$port = if ($env:VISION_SERVE_PORT) { [int]$env:VISION_SERVE_PORT } else { 8100 }
$ckpt = $env:REACTION_CHECKPOINT
if (-not $ckpt) {
    $ckpt = "C:\Users\zakar\homelab-data\checkpoints\8117bfe1-85ea-4074-a967-5ebeedd40698\best_model.pt"
}
$env:REACTION_CHECKPOINT = $ckpt
Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 2
$logDir = "C:\Users\zakar\homelab-data\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
Start-Process powershell.exe -ArgumentList @(
    "-NoProfile", "-ExecutionPolicy", "Bypass",
    "-File", "C:\Users\zakar\homelab-gpu\scripts\start-reaction-serve.ps1"
) -RedirectStandardOutput "$logDir\reaction-serve.log" `
  -RedirectStandardError "$logDir\reaction-serve.err" -WindowStyle Hidden
Write-Host "Restarted on :$port with $ckpt"
