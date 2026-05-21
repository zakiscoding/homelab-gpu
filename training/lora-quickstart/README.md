# LoRA quickstart (RTX 5070 12GB)

First fine-tune project placeholder. Use **QLoRA** — full fine-tune usually does not fit 12GB VRAM.

## Prerequisites

1. `homelab-gpu` cloned to `C:\Users\zakar\homelab-gpu`
2. `.env` from `.env.example` — set `CHECKPOINT_DIR`, `DATASET_DIR`, optional `HF_TOKEN`
3. `.\scripts\setup-venv.ps1` then add training stack to `requirements.txt` (e.g. Unsloth or `torch` + `peft` + `transformers`)
4. `.\scripts\stop-ollama.ps1` before training

## Data layout (local disk, not git)

```text
C:\Users\zakar\homelab-data\
├── datasets\
│   └── my-task\
│       └── train.jsonl
└── checkpoints\
    └── my-task-lora\
```

**Jupyter:** use [notebooks/01_gpu_and_paths.ipynb](../../notebooks/01_gpu_and_paths.ipynb) — start with `.\scripts\start-jupyter.ps1`

## Next steps (pick a stack)

| Stack | Notes |
|-------|--------|
| [Unsloth](https://github.com/unslothai/unsloth) | Good for LoRA on consumer GPUs |
| LLaMA-Factory | Often via WSL2 on Windows |
| Hugging Face PEFT + transformers | Manual scripts in this folder |

Add `train.py` here when you choose a stack; log the run in [homelab SETUP-LOG](https://github.com/zakiscoding/homelab/blob/main/docs/SETUP-LOG.md).

## After training

Export adapter → merge → GGUF (tool-dependent) → load in Ollama on `zak_pc` for inference.
