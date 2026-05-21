# Notebooks (GPU PC)

Interactive fine-tuning and experiments on **`zak_pc`**.

| Notebook | Purpose |
|----------|---------|
| [01_gpu_and_paths.ipynb](01_gpu_and_paths.ipynb) | CUDA check, `CHECKPOINT_DIR` / `DATASET_DIR` |
| `02_lora_experiment.ipynb` | *(you create)* — dataset + LoRA training |

## Start JupyterLab

From repo root:

```powershell
cd C:\Users\zakar\homelab-gpu
.\scripts\setup-venv.ps1    # first time only
.\scripts\start-jupyter.ps1
```

Open the URL printed in the terminal (default `http://127.0.0.1:8888`).

### Remote from laptop (`cashh`) over Tailscale

1. On GPU PC, allow port **8888** (Private firewall) if needed.
2. Start with LAN bind:
   ```powershell
   $env:JUPYTER_LISTEN = "0.0.0.0"
   .\scripts\start-jupyter.ps1
   ```
3. On laptop browser: `http://100.88.71.33:8888` (use token from GPU terminal output).

**Security:** Do not expose 8888 to the public internet without a strong Jupyter token.

## Kernel

Uses `.venv` from repo root. If Jupyter does not see the venv:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name homelab-gpu --display-name "Python 3 (homelab-gpu)"
```
