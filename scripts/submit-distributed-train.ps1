# Distributed CNN train: iMac shards batches (train_prep) → GPU trains with AMP.
param(
    [string]$ApiUrl = "http://100.91.113.98:3080/api/homelab",
    [string]$ApiKey = $env:HOMELAB_API_KEY,
    [ValidateSet("cifar10", "fer2013")]
    [string]$Dataset = "cifar10",
    [string]$ModelVersion = "reaction-dist-v1",
    [int]$ShardCount = 8,
    [int]$Epochs = 5,
    [int]$BatchSize = 32,
    [int]$GradAccum = 4,
    [double]$ValRatio = -1,
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
if (-not $ApiKey) { throw "Set API_KEY" }

if ($ValRatio -lt 0) { $ValRatio = if ($Dataset -eq "fer2013") { 0 } else { 0.1 } }

$body = @{
    train_mode       = "cnn"
    dataset          = $Dataset
    dataset_version  = $Dataset
    model_version    = $ModelVersion
    architecture     = "resnet18"
    distributed      = $true
    shard_count      = $ShardCount
    val_ratio        = $ValRatio
    epochs           = $Epochs
    batch_size       = $BatchSize
    grad_accum_steps = $GradAccum
    use_amp          = $true
}
if ($Dataset -eq "fer2013") {
    $body.labels = "ferplus"
    $body.architecture = "resnet34"
    $body.accurate_aug = $true
    $body.class_weights = $true
    $body.label_smoothing = 0.1
    $body.learning_rate = 3e-4
}
$bodyJson = $body | ConvertTo-Json -Compress

$h = @{ "x-api-key" = $ApiKey; "Content-Type" = "application/json" }
$job = Invoke-RestMethod -Uri "$ApiUrl/jobs" -Method Post -Headers $h -Body $bodyJson
Write-Host "Train job $($job.id) — iMac will run train_prep ($ShardCount shards), then GPU claims train."
Write-Host "Ensure imac-worker + GPU worker are running."

if ($Wait) {
    do {
        Start-Sleep -Seconds 20
        $j = Invoke-RestMethod -Uri "$ApiUrl/jobs/$($job.id)" -Headers @{ "x-api-key" = $ApiKey }
        Write-Host "  train $($j.status)"
    } while ($j.status -in @("pending", "running"))
}
