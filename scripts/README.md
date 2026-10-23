# Scripts

[Back to homelab-gpu README](../README.md)

**PowerShell helpers for the GPU PC on Windows (`zak_pc`, user `zakar`).** Run them from the repo root (`C:\Users\zakar\homelab-gpu`). Scripts that need a config load `.env` from the repo root.

## Setup and checks

| Script | Purpose |
|--------|---------|
| `setup-venv.ps1` | Create `.venv` and `pip install -r requirements.txt` |
| `verify-gpu.ps1` | `nvidia-smi`, plus a torch CUDA check if `.venv` exists |
| `test_cuda.py` | Print Python, torch CUDA device, and bitsandbytes versions |

## Ollama

| Script | Purpose |
|--------|---------|
| `install-ollama-serve-task.ps1` | **Administrator, once.** Sets machine `OLLAMA_HOST=0.0.0.0:11434`, registers the `HomelabOllamaServe` task (at logon, restarts on failure), adds the "Ollama LAN" firewall rule |
| `start-ollama.ps1` | Run the task (or start `ollama serve` directly), then check the port and list models |
| `serve-ollama.bat` | Set `OLLAMA_HOST` and run `ollama serve` in a console |
| `stop-ollama.ps1` | Stop Ollama to free VRAM, then show `nvidia-smi`. The QLoRA trainer calls it automatically |

## Train worker

| Script | Purpose |
|--------|---------|
| `start-gpu-worker.ps1` | Run `python -m worker.poll_jobs` in the foreground |
| `start-gpu-worker-bg.ps1` | Start the worker as a hidden background process (used by the scheduled task) |
| `register-gpu-worker-task.ps1` | **Administrator.** Register `HomelabGpuTrainWorker` (at logon, runs `start-gpu-worker-bg.ps1`) |
| `run-worker-once.ps1` | One poll cycle (`--once`) in the console, for debugging |
| `run-worker-detached.ps1` | One poll cycle that survives SSH disconnect. Logs to `C:\Users\zakar\homelab-data\worker-last.log` / `.err` |

## Training

| Script | Purpose |
|--------|---------|
| `train_once_job.py` | `python scripts\train_once_job.py <job_id> [payload.json]`: QLoRA for one job id, without the queue |
| `run-train-detached.ps1` | `-JobId <uuid>`: run `train_once_job.py` detached. Logs to `C:\Users\zakar\homelab-data\train-last.log` / `.err` |
| `sync-adapter-to-imac.ps1` | `-JobId <uuid> [-ModelVersion name] [-ImacHost imac-ts]`: scp an adapter to the iMac ML-STORE |

## Metrics and notebooks

| Script | Purpose |
|--------|---------|
| `push-gpu-metrics.ps1` | Push utilization, memory, and temperature from `nvidia-smi` to the iMac Pushgateway (`PUSHGATEWAY_URL`) |
| `register-gpu-metrics-task.ps1` | Register `HomelabGpuMetrics` (runs `push-gpu-metrics.ps1` every 1 min, through `run-hidden-powershell.vbs` so no console flashes) |
| `run-hidden-powershell.vbs` | Run a `.ps1` with no visible window. Scheduled tasks launch PowerShell through it |
| `fix-console-popups.ps1` | **Administrator.** Re-register `HomelabGpuMetrics` and `HomelabOllamaServe` as hidden tasks and disable the duplicate `OllamaServe` task |
| `hide-homelab-tasks.ps1` | Set the Hidden flag on existing homelab tasks. Lighter than `fix-console-popups.ps1`, but PowerShell can still flash |
| `start-jupyter.ps1` | JupyterLab on `127.0.0.1:8888` (`JUPYTER_LISTEN` and `JUPYTER_PORT` override). See [notebooks](../notebooks/README.md) |

## Common commands

```powershell
cd C:\Users\zakar\homelab-gpu
.\scripts\setup-venv.ps1
.\scripts\verify-gpu.ps1
.\scripts\start-gpu-worker.ps1
.\scripts\run-train-detached.ps1 -JobId <uuid>
```
