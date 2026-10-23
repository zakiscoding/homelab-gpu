# Notebooks

[Back to homelab-gpu README](../README.md)

**Interactive fine-tuning and experiments on the GPU PC (`zak_pc`, Windows boot).**

| Notebook | Purpose |
|----------|---------|
| [01_gpu_and_paths.ipynb](01_gpu_and_paths.ipynb) | CUDA check, `CHECKPOINT_DIR` / `DATASET_DIR` |
| `02_lora_experiment.ipynb` | Not created yet: dataset + LoRA training |

## Start JupyterLab

From the repo root:

```powershell
cd C:\Users\zakar\homelab-gpu
.\scripts\setup-venv.ps1      # first time only
.\scripts\stop-ollama.ps1     # only if you will train (frees VRAM)
.\scripts\start-jupyter.ps1
```

Open the URL printed in the terminal (default `http://127.0.0.1:8888`). `start-jupyter.ps1` loads `.env`, so the notebook sees `CHECKPOINT_DIR` and `DATASET_DIR`.

## Remote from the laptop over Tailscale

1. On the GPU PC, allow port 8888 in the Private firewall profile if needed.
2. Start with a LAN bind:
   ```powershell
   $env:JUPYTER_LISTEN = "0.0.0.0"
   .\scripts\start-jupyter.ps1
   ```
3. In the laptop browser, open `http://100.88.71.33:8888` and use the token from the GPU terminal output.

Do not expose 8888 to the public internet. Keep it on the tailnet and use the Jupyter token.

## Kernel

Uses `.venv` from the repo root. If Jupyter does not see the venv:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name homelab-gpu --display-name "Python 3 (homelab-gpu)"
```
