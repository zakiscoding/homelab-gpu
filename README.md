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
├── scripts/              # PowerShell helpers (env, launch, stop Ollama before train)
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

## Before a long training run

Free VRAM for training — reduce Ollama load on the same GPU:

```powershell
# See homelab/docs/nodes/gpu-node.md
Stop-Process -Name ollama* -Force -ErrorAction SilentlyContinue
nvidia-smi
```

Restart Ollama after training when you need inference again.

## Related repos

| Repo | Purpose |
|------|---------|
| [homelab](https://github.com/zakiscoding/homelab) | Documentation |
| [homelab-services](https://github.com/zakiscoding/homelab-services) | iMac API / DB stack |
| **homelab-gpu** | This repo — training code on GPU PC |
