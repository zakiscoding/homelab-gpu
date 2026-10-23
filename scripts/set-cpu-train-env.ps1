# Tune i7-14700F (or similar) for GPU training: parallel DataLoader + no thread oversubscription.
# Dot-source before train:  . .\scripts\set-cpu-train-env.ps1
$logical = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
$workers = [Math]::Max(4, [Math]::Min(12, [int]($logical / 2)))
$env:CNN_NUM_WORKERS = "$workers"
$env:CNN_PREFETCH = "4"
$env:TRAIN_MAIN_THREADS = "4"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
Write-Host "CPU train env: CNN_NUM_WORKERS=$workers (logical CPUs=$logical)"
