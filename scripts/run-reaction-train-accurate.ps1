# Max-accuracy FER training: FER+ labels, ResNet34, strong aug, class weights
param(
    [int]$Epochs = 60,
    [int]$BatchSize = 48
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path (Split-Path $MyInvocation.MyCommand.Path -Parent) -Parent
. (Join-Path $Repo "scripts\set-cpu-train-env.ps1")
$Venv = Join-Path $Repo ".venv\Scripts\python.exe"
if (-not (Test-Path $Venv)) { $Venv = "python" }

& $Venv "$Repo\scripts\fetch_fer2013_csv.py"
& $Venv "$Repo\scripts\fetch_ferplus.py"

& $Venv "$Repo\scripts\train_reaction_gpu.py" `
    --dataset fer2013 `
    --labels ferplus `
    --arch resnet34 `
    --epochs $Epochs `
    --batch-size $BatchSize `
    --grad-accum 4 `
    --lr 3e-4 `
    --label-smoothing 0.1 `
    --class-weights `
    --accurate-aug
