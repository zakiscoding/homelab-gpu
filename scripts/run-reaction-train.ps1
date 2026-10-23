# GPU-only facial expression training (FER2013 or personal folders)
param(
    [ValidateSet("fer2013", "cifar10", "reactions")]
    [string]$Dataset = "fer2013",
    [string]$DatasetVersion = "me-reactions",
    [int]$Epochs = 15,
    [int]$BatchSize = 64
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path (Split-Path $MyInvocation.MyCommand.Path -Parent) -Parent
$Venv = Join-Path $Repo ".venv\Scripts\python.exe"
if (-not (Test-Path $Venv)) { $Venv = "python" }

if ($Dataset -eq "fer2013") {
    & $Venv "$Repo\scripts\fetch_fer2013_kagglehub.py"
    if ($LASTEXITCODE -ne 0) {
        & "$PSScriptRoot\download-fer2013.ps1"
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}

$dsArg = if ($Dataset -eq "reactions") { "folder" } else { $Dataset }
& $Venv "$Repo\scripts\train_reaction_gpu.py" `
    --dataset $dsArg `
    --dataset-version $DatasetVersion `
    --epochs $Epochs `
    --batch-size $BatchSize
