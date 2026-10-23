# Distributed FER/reaction train: iMac shards indices (CPU) → GPU trains (CUDA).
# Uses official FER2013 PublicTest for val (val_ratio=0 in plan).
param(
    [string]$ApiUrl = "http://100.91.113.98:3080/api/homelab",
    [string]$ApiKey = $env:HOMELAB_API_KEY,
    [ValidateSet("fer", "ferplus")]
    [string]$Labels = "ferplus",
    [ValidateSet("resnet18", "resnet34")]
    [string]$Architecture = "resnet34",
    [int]$ShardCount = 8,
    [int]$Epochs = 60,
    [int]$BatchSize = 48,
    [int]$GradAccum = 4,
    [switch]$Wait
)
$ErrorActionPreference = "Stop"
if (-not $ApiKey) {
    $envFile = "C:\Users\cashh\homelab-services\.env"
    if (Test-Path $envFile) {
        Get-Content $envFile | ForEach-Object {
            if ($_ -match '^\s*API_KEY=(.+)$') { $ApiKey = $matches[1].Trim() }
        }
    }
}
if (-not $ApiKey) { throw "Set HOMELAB_API_KEY or API_KEY in homelab-services\.env" }

$body = @{
    train_mode       = "cnn"
    dataset          = "fer2013"
    model_version    = "reaction-dist-ferplus-v1"
    architecture     = $Architecture
    labels           = $Labels
    distributed      = $true
    shard_count      = $ShardCount
    val_ratio        = 0
    epochs           = $Epochs
    batch_size       = $BatchSize
    grad_accum_steps = $GradAccum
    use_amp          = $true
    learning_rate    = 3e-4
    label_smoothing  = 0.1
    class_weights    = $true
    accurate_aug     = $true
    num_workers      = 10
} | ConvertTo-Json -Compress

$h = @{ "x-api-key" = $ApiKey; "Content-Type" = "application/json" }
$job = Invoke-RestMethod -Uri "$ApiUrl/jobs" -Method Post -Headers $h -Body $body
Write-Host "Reaction train job $($job.id)"
Write-Host "  iMac: train_prep ($ShardCount shards, val_ratio=0 → GPU uses FER PublicTest)"
Write-Host "  GPU:  claims train when plan ready (ensure imac-worker + GPU worker running)"
Write-Host "Prereq on zak_pc: fer2013.csv + fer2013new.csv under homelab-data\datasets\fer2013\"

if ($Wait) {
    do {
        Start-Sleep -Seconds 30
        $j = Invoke-RestMethod -Uri "$ApiUrl/jobs/$($job.id)" -Headers @{ "x-api-key" = $ApiKey }
        Write-Host "  $($j.status)"
    } while ($j.status -in @("pending", "running"))
    if ($j.status -eq "succeeded") { Write-Host "Done. Checkpoint on GPU under checkpoints\$($job.id)" }
}
