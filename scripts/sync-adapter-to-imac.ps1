# Copy adapter from GPU checkpoint to iMac ML-STORE (hybrid storage).
# Usage: .\scripts\sync-adapter-to-imac.ps1 -JobId <uuid> -ModelVersion "my-lora-v1"
param(
    [Parameter(Mandatory = $true)][string]$JobId,
    [string]$ModelVersion = "",
    [string]$ImacHost = "imac-ts",
    [string]$MlStoreRemote = "/Volumes/1tb drive/ML-STORE/models/adapters"
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$src = Join-Path $env:CHECKPOINT_DIR "..\..\homelab-data\checkpoints\$JobId\adapter"
if (-not $env:CHECKPOINT_DIR) { $src = "C:\Users\zakar\homelab-data\checkpoints\$JobId\adapter" }
if (-not (Test-Path $src)) { $src = "C:\Users\zakar\homelab-data\checkpoints\$JobId\adapter" }
if (-not (Test-Path $src)) { throw "Adapter not found: $src" }
$ver = if ($ModelVersion) { $ModelVersion } else { "adapter-$($JobId.Substring(0,8))" }
$dest = "$MlStoreRemote/$ver"
Write-Host "Sync $src -> ${ImacHost}:$dest"
scp -r "$src" "${ImacHost}:$dest"
Write-Host "Done. iMac path: $dest"
