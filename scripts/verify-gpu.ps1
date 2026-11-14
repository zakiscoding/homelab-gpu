# Quick GPU + Python check before training
Write-Host "=== nvidia-smi ==="
nvidia-smi
Write-Host "`n=== Python ==="
if (Get-Command python -ErrorAction SilentlyContinue) {
    python --version
    if (Test-Path ".\.venv\Scripts\python.exe") {
        .\.venv\Scripts\python.exe -c "import sys; print('venv:', sys.executable)"
        .\.venv\Scripts\python.exe -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')" 2>$null
        if ($LASTEXITCODE -ne 0) { Write-Host "torch not installed yet - run setup-venv.ps1 after pinning requirements.txt" }
    } else {
        Write-Host "No .venv - run scripts\setup-venv.ps1"
    }
} else {
    Write-Host "Python not installed"
}
