param(
    [Parameter(Mandatory = $true)]
    [string]$CheckpointDir
)

$Repo = Split-Path (Split-Path $MyInvocation.MyCommand.Path -Parent) -Parent
$Venv = Join-Path $Repo ".venv\Scripts\python.exe"
$ckpt = Join-Path $CheckpointDir "best_model.pt"
if (-not (Test-Path $ckpt)) { throw "Missing $ckpt" }
& $Venv "$Repo\scripts\infer_reaction_webcam.py" --checkpoint $ckpt
