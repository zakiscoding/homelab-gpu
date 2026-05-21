# Free GPU VRAM before training. Restart Ollama from tray or: Start-Process ollama serve
Stop-Process -Name ollama* -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
nvidia-smi
