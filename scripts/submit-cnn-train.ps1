# Submit a CNN train job to the iMac API (run from laptop or GPU).
param(
    [string]$ApiUrl = "http://100.91.113.98:3080/api/homelab",
    [string]$ApiKey = $env:HOMELAB_API_KEY,
    [string]$Dataset = "cifar10",
    [string]$DatasetVersion = "cifar10",
    [string]$ModelVersion = "cifar-resnet18-v1",
    [string]$Architecture = "resnet18",
    [int]$Epochs = 5,
    [int]$BatchSize = 32,
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
if (-not $ApiKey) { throw "Set -ApiKey or HOMELAB_API_KEY or homelab-services/.env API_KEY" }

$body = @{
    train_mode      = "cnn"
    dataset         = $Dataset
    dataset_version = $DatasetVersion
    model_version   = $ModelVersion
    architecture    = $Architecture
    epochs          = $Epochs
    batch_size      = $BatchSize
} | ConvertTo-Json -Compress

$h = @{ "x-api-key" = $ApiKey; "Content-Type" = "application/json" }
$job = Invoke-RestMethod -Uri "$ApiUrl/jobs" -Method Post -Headers $h -Body $body
Write-Host "Created job $($job.id) status=$($job.status)"
Write-Host "Poll: Invoke-RestMethod -Uri '$ApiUrl/jobs/$($job.id)' -Headers @{ 'x-api-key' = '...' }"

if ($Wait) {
    do {
        Start-Sleep -Seconds 15
        $j = Invoke-RestMethod -Uri "$ApiUrl/jobs/$($job.id)" -Headers @{ "x-api-key" = $ApiKey }
        Write-Host "  status=$($j.status)"
    } while ($j.status -in @("pending", "running"))
    if ($j.status -eq "succeeded") {
        Write-Host "Done. Weights on GPU: homelab-data\checkpoints\$($job.id)\best_model.pt"
    } else {
        Write-Host "Failed: $($j.error_message)"
    }
}
