# Start ResNet34 FER training in background on zak_pc
$Repo = "C:\Users\zakar\homelab-gpu"
$Py = "$Repo\.venv\Scripts\python.exe"
$Log = "C:\Users\zakar\homelab-data\logs\reaction-accurate.log"
$Err = "C:\Users\zakar\homelab-data\logs\reaction-accurate.err"
New-Item -ItemType Directory -Force -Path (Split-Path $Log) | Out-Null
$env:CNN_NUM_WORKERS = "10"
$env:CNN_PREFETCH = "4"
$env:TRAIN_MAIN_THREADS = "4"
$env:OMP_NUM_THREADS = "1"
$args = @(
    "-u", "scripts\train_reaction_gpu.py",
    "--dataset", "folder",
    "--dataset-version", "fer2013",
    "--arch", "resnet34",
    "--epochs", "60",
    "--batch-size", "48",
    "--grad-accum", "4",
    "--lr", "3e-4",
    "--label-smoothing", "0.1",
    "--class-weights",
    "--accurate-aug"
)
Start-Process -FilePath $Py -ArgumentList $args -WorkingDirectory $Repo `
    -WindowStyle Hidden -RedirectStandardOutput $Log -RedirectStandardError $Err
Write-Host "Started. Log: $Log"
