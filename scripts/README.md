# Scripts (GPU PC — `zak_pc`)

| Script | Purpose |
|--------|---------|
| `stop-ollama.ps1` | Stop Ollama to free VRAM before training |
| `setup-venv.ps1` | Create `.venv` and `pip install -r requirements.txt` |
| `verify-gpu.ps1` | `nvidia-smi` + optional torch CUDA check |

Run from repo root:

```powershell
cd C:\Users\zakar\homelab-gpu
.\scripts\setup-venv.ps1
.\scripts\verify-gpu.ps1
```
