# homelab-gpu

**GPU PC training workspace** — fine-tuning scripts, configs, and launch helpers for `zak_pc`.

| | |
|--|--|
| **GitHub** | [github.com/zakiscoding/homelab-gpu](https://github.com/zakiscoding/homelab-gpu) (private) |
| **Runs on** | Windows GPU PC (`zak_pc`, user **`zakar`**) |
| **Inference** | Ollama on `:11434` — documented in [homelab](https://github.com/zakiscoding/homelab), not this repo |
| **Docs** | Architecture, setup log, GPU runbook → `zakiscoding/homelab` |

## Roles on the GPU PC

| Role | Where | This repo? |
|------|-------|------------|
| **Inference** | Ollama native | No — see `homelab/docs/nodes/gpu-node.md` |
| **Training** | Python / PyTorch / LoRA tooling | **Yes** |

## Layout

```text
homelab-gpu/
├── README.md
├── .gitignore
├── requirements.txt      # Python deps for training (pin as you add tools)
├── configs/              # YAML/JSON training configs (small files only)
├── scripts/              # PowerShell helpers (env, launch, stop Ollama, Jupyter, GPU worker)
├── worker/               # Phase 0A — poll_jobs.py, run_train.py (MVW loop → iMac API)
├── notebooks/            # JupyterLab — GPU check, LoRA experiments
└── training/             # Training entrypoints (add per project)
```

## Do not commit

- Model weights / checkpoints (GB–TB)
- Datasets
- `venv/`, `.venv/`
- Ollama models (`~/.ollama` or Windows equivalent)
- Secrets (`.env`, API keys)

Keep artifacts on **GPU local disk** (e.g. `D:\homelab-data\` — document path in [homelab SETUP-LOG](https://github.com/zakiscoding/homelab/blob/main/docs/SETUP-LOG.md)).

## Clone (GPU PC — Windows)

```powershell
git clone https://github.com/zakiscoding/homelab-gpu.git C:\Users\zakar\homelab-gpu
cd C:\Users\zakar\homelab-gpu
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Ollama (inference, LAN / Tailscale)

Ollama must listen on **`0.0.0.0:11434`** so the iMac API and web UI can reach it.

| Script | Purpose |
|--------|---------|
| `scripts/install-ollama-serve-task.ps1` | **Admin** — scheduled task `HomelabOllamaServe` + firewall |
| `scripts/serve-ollama.bat` | `OLLAMA_HOST` + `ollama serve` |
| `scripts/start-ollama.ps1` | Quick start / run existing task |
| `scripts/stop-ollama.ps1` | Stop Ollama before training (free VRAM) |

```powershell
# First time (Admin)
.\scripts\install-ollama-serve-task.ps1

# If chat hangs in the web UI
schtasks /Run /TN HomelabOllamaServe
curl.exe http://100.88.71.33:11434/api/tags
```

Full runbook: [homelab gpu-node.md](https://github.com/zakiscoding/homelab/blob/main/docs/nodes/gpu-node.md).

## Reaction vision (FER CNN, port 8100)

Separate from Ollama — serves `POST /infer/reaction` for the homelab-web **Reactions** page.

| Item | Detail |
|------|--------|
| Port | `8100` on `0.0.0.0` (Tailscale `100.88.71.33`) |
| iMac proxy | `GPU_VISION_URL` in homelab-services `.env` |
| Docs | [homelab vision-reactions.md](https://github.com/zakiscoding/homelab/blob/main/docs/services/vision-reactions.md) |

```powershell
# When scripts are present in this repo:
.\scripts\start-reaction-serve-bg.ps1
curl.exe http://127.0.0.1:8100/health
```

Stop Ollama first if both compete for VRAM on the same GPU.

## Phase 0A GPU worker

Polls iMac for `train` jobs (Postgres control plane). Docs: [homelab PHASE-0A](https://github.com/zakiscoding/homelab/blob/main/docs/PHASE-0A.md).

```powershell
# .env: API_BASE_URL=http://100.91.113.98, WORKER_API_KEY=<match iMac .env>
.\scripts\start-gpu-worker.ps1
```

RTX 5070: stub train in 0A does not use CUDA; real LoRA needs `stop-ollama` first.

## Before a long training run

Free VRAM for training — reduce Ollama load on the same GPU:

```powershell
# See homelab/docs/nodes/gpu-node.md
Stop-Process -Name ollama* -Force -ErrorAction SilentlyContinue
nvidia-smi
```

Restart Ollama after training when you need inference again:

```powershell
schtasks /Run /TN HomelabOllamaServe
# or: .\scripts\start-ollama.ps1
```

## Related repos

| Repo | Purpose |
|------|---------|
| [homelab](https://github.com/zakiscoding/homelab) | Documentation |
| [homelab-services](https://github.com/zakiscoding/homelab-services) | iMac API / DB stack |
| **homelab-gpu** | This repo — training code on GPU PC |
