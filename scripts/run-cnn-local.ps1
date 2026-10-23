# GPU-only CNN train (optimized). No iMac Postgres required.
param(
    [string]$Dataset = "cifar10",
    [string]$DatasetVersion = "",
    [int]$Epochs = 10,
    [int]$BatchSize = 32,
    [int]$GradAccum = 4,
    [string]$ModelVersion = "gpu-cnn-v1"
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$py = Join-Path $Repo ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Run .\scripts\setup-venv.ps1 first" }

Write-Host "Stopping Ollama to free VRAM..."
& (Join-Path $Repo "scripts\stop-ollama.ps1")

$min = Join-Path $Repo "scripts\train_cifar_gpu_min.py"
if ($Dataset -eq "cifar10" -and (Test-Path $min)) {
    $env:CNN_EPOCHS = "$Epochs"
    $env:CNN_BATCH_SIZE = "$BatchSize"
    $env:CNN_GRAD_ACCUM = "$GradAccum"
    Push-Location $Repo
    try { & $py $min } finally { Pop-Location }
    return
}

$args = @(
    (Join-Path $Repo "scripts\run_cnn_local.py"),
    "--dataset", $Dataset,
    "--epochs", $Epochs,
    "--batch-size", $BatchSize,
    "--grad-accum", $GradAccum,
    "--model-version", $ModelVersion
)
if ($DatasetVersion) { $args += @("--dataset-version", $DatasetVersion) }

Push-Location $Repo
try {
    & $py @args
} finally {
    Pop-Location
}
