# Download FER2013 CSV into homelab-data/datasets/fer2013/
# Sources (pick one):
#   1. Kaggle: kaggle datasets download -d msambare/fer2013 -p $OutDir --unzip
#   2. Manual: copy fer2013.csv to $OutDir\fer2013.csv

$ErrorActionPreference = "Stop"
$OutDir = if ($env:DATASET_DIR) { Join-Path $env:DATASET_DIR "fer2013" } else { "C:\Users\zakar\homelab-data\datasets\fer2013" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$Csv = Join-Path $OutDir "fer2013.csv"

if (Test-Path $Csv) {
    Write-Host "Already exists: $Csv"
    exit 0
}

$Py = Join-Path (Split-Path $PSScriptRoot -Parent) ".venv\Scripts\python.exe"
if (Test-Path $Py) {
    & $Py "$PSScriptRoot\fetch_fer2013_kagglehub.py"
    if ($LASTEXITCODE -eq 0) { exit 0 }
}

if (Get-Command kaggle -ErrorAction SilentlyContinue) {
    Write-Host "Downloading via Kaggle CLI..."
    Push-Location $OutDir
    kaggle datasets download -d msambare/fer2013 --unzip
    Pop-Location
    if (-not (Test-Path $Csv)) {
        $found = Get-ChildItem -Path $OutDir -Filter "fer2013.csv" -Recurse | Select-Object -First 1
        if ($found) { Copy-Item $found.FullName $Csv -Force }
    }
}

if (-not (Test-Path $Csv)) {
    Write-Host @"

FER2013 not found at: $Csv

Options:
  A) Install Kaggle CLI, set KAGGLE_USERNAME / KAGGLE_KEY, re-run this script
  B) Download from https://www.kaggle.com/datasets/msambare/fer2013
     and place fer2013.csv at: $Csv

Then run: .\scripts\run-reaction-train.ps1 -Dataset fer2013
"@
    exit 1
}

Write-Host "Ready: $Csv"
